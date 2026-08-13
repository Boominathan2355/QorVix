#include "qorvix/vision/preprocess.hpp"

#include <algorithm>
#include <cmath>

namespace qorvix::vision {

namespace {

// PIL's bicubic kernel (Pillow's src/libImaging/Resample.c), a = -0.5. Note this is Catmull-Rom,
// NOT the a = -0.75 that OpenCV uses — the two differ enough to move an embedding, and CLIP
// references are produced through PIL.
float bicubicKernel(float x) {
  constexpr float a = -0.5f;
  x = std::abs(x);
  if (x < 1.0f) return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
  if (x < 2.0f) return (((x - 5.0f) * x + 8.0f) * x - 4.0f) * a;
  return 0.0f;
}

// Resamples one axis with the filter support scaled by the downsampling factor. The scaling is
// what makes shrinking an average over the whole source span rather than a 4-tap point sample;
// without it a 1000px photo reduced to 336px aliases visibly and the vector drifts.
//
// The window bounds follow Pillow's Resample.c exactly:
//   xmin = (int)(center - support + 0.5)
//   xmax = (int)(center + support + 0.5)
// both TRUNCATED, not floor/ceil. Using ceil for the upper bound pulls in an extra source pixel,
// which is invisible on smooth gradients and badly wrong on high-frequency detail — measured at
// 1.8e-1 on a checkerboard against 6e-3 elsewhere.
void resampleAxis(const std::vector<float>& src, int srcW, int srcH, int dstW,
                  std::vector<float>& dst) {
  dst.assign(static_cast<std::size_t>(dstW) * srcH * 3, 0.0f);
  const double scale = static_cast<double>(srcW) / dstW;
  const double filterScale = std::max(1.0, scale);
  const double support = 2.0 * filterScale;  // bicubic support is 2

  for (int x = 0; x < dstW; ++x) {
    const double center = (x + 0.5) * scale;
    int left = static_cast<int>(center - support + 0.5);
    int right = static_cast<int>(center + support + 0.5);
    left = std::max(left, 0);
    right = std::min(right, srcW);
    if (right <= left) {
      left = std::min(std::max(0, static_cast<int>(center)), srcW - 1);
      right = left + 1;
    }

    std::vector<float> weights(static_cast<std::size_t>(right - left));
    double total = 0.0;
    for (int s = left; s < right; ++s) {
      const double w = bicubicKernel(static_cast<float>((s + 0.5 - center) / filterScale));
      weights[static_cast<std::size_t>(s - left)] = static_cast<float>(w);
      total += w;
    }
    // Normalize per output pixel: the truncated support at the image edges does not sum to 1, and
    // skipping this darkens or brightens the border rows.
    if (total != 0.0) {
      for (auto& w : weights) w = static_cast<float>(w / total);
    }

    for (int y = 0; y < srcH; ++y) {
      float acc[3] = {0.0f, 0.0f, 0.0f};
      for (int s = left; s < right; ++s) {
        const float w = weights[static_cast<std::size_t>(s - left)];
        const float* p = src.data() + (static_cast<std::size_t>(y) * srcW + s) * 3;
        acc[0] += w * p[0];
        acc[1] += w * p[1];
        acc[2] += w * p[2];
      }
      float* d = dst.data() + (static_cast<std::size_t>(y) * dstW + x) * 3;
      d[0] = acc[0];
      d[1] = acc[1];
      d[2] = acc[2];
    }
  }
}

// Transposes an HWC float image so the same one-axis resampler can handle the vertical pass.
void transpose(const std::vector<float>& src, int w, int h, std::vector<float>& dst) {
  dst.assign(static_cast<std::size_t>(w) * h * 3, 0.0f);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float* s = src.data() + (static_cast<std::size_t>(y) * w + x) * 3;
      float* d = dst.data() + (static_cast<std::size_t>(x) * h + y) * 3;
      d[0] = s[0];
      d[1] = s[1];
      d[2] = s[2];
    }
  }
}

// Rounds an intermediate pass back to 8-bit. Pillow resamples 8-bit images with fixed-point
// coefficients and clips each pass to uint8 before the next one runs, so a float-throughout
// pipeline drifts by roughly half a level per pass — about 6e-3 in CLIP's normalized units, on
// every pixel. Matching the reference means quantizing where the reference quantizes.
void quantizeToByte(std::vector<float>& v) {
  for (float& x : v) x = static_cast<float>(std::clamp(std::lround(x), 0L, 255L));
}

bool resizeFloat(const std::vector<float>& src, int srcW, int srcH, int dstW, int dstH,
                 std::vector<float>& out) {
  if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0) return false;

  // Horizontal pass, then transpose + horizontal pass again = vertical. Two 1-D passes rather
  // than one 2-D kernel, which is both faster and what Pillow does. Each pass is skipped when
  // that axis is unchanged — Pillow skips it too, and running a unit-scale filter anyway would
  // blur an image that should have been copied through.
  std::vector<float> horiz;
  if (dstW != srcW) {
    resampleAxis(src, srcW, srcH, dstW, horiz);
    quantizeToByte(horiz);
  } else {
    horiz = src;
  }

  if (dstH == srcH) {
    out = std::move(horiz);
    return true;
  }

  std::vector<float> t;
  transpose(horiz, dstW, srcH, t);  // now [srcH wide, dstW tall]

  std::vector<float> vert;
  resampleAxis(t, srcH, dstW, dstH, vert);
  quantizeToByte(vert);

  transpose(vert, dstH, dstW, out);  // back to [dstW, dstH]
  return true;
}

}  // namespace

bool resizeBicubic(const Image& src, int dstW, int dstH, Image& out, std::string& error) {
  if (!src.valid()) {
    error = "invalid source image";
    return false;
  }
  std::vector<float> in(static_cast<std::size_t>(src.width) * src.height * 3);
  for (std::size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(src.rgb[i]);

  std::vector<float> resized;
  if (!resizeFloat(in, src.width, src.height, dstW, dstH, resized)) {
    error = "invalid resize target";
    return false;
  }
  out.width = dstW;
  out.height = dstH;
  out.rgb.assign(resized.size(), 0);
  for (std::size_t i = 0; i < resized.size(); ++i) {
    out.rgb[i] = static_cast<std::uint8_t>(std::clamp(std::lround(resized[i]), 0L, 255L));
  }
  return true;
}

bool preprocessClip(const Image& img, const PreprocessConfig& cfg, std::vector<float>& out,
                    std::string& error) {
  error.clear();
  if (!img.valid()) {
    error = "invalid image";
    return false;
  }
  if (cfg.size <= 0) {
    error = "invalid target size";
    return false;
  }

  // 1. Resize the SHORTEST side to cfg.size, preserving aspect ratio.
  const int shortest = std::min(img.width, img.height);
  const double scale = static_cast<double>(cfg.size) / shortest;
  const int rw = std::max(1, static_cast<int>(std::round(img.width * scale)));
  const int rh = std::max(1, static_cast<int>(std::round(img.height * scale)));

  std::vector<float> in(static_cast<std::size_t>(img.width) * img.height * 3);
  for (std::size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(img.rgb[i]);

  std::vector<float> resized;
  if (!resizeFloat(in, img.width, img.height, rw, rh, resized)) {
    error = "resize failed";
    return false;
  }

  // 2. Centre crop. The origin uses integer division exactly as CLIPImageProcessor does; rounding
  // the other way shifts the crop by a pixel, which is small but not nothing at 336px.
  const int x0 = (rw - cfg.size) / 2;
  const int y0 = (rh - cfg.size) / 2;

  // 3+4. Rescale to [0,1] and normalize, writing CHW directly.
  const std::size_t plane = static_cast<std::size_t>(cfg.size) * cfg.size;
  out.assign(plane * 3, 0.0f);
  for (int c = 0; c < 3; ++c) {
    const float mean = cfg.mean[static_cast<std::size_t>(c)];
    const float sd = cfg.std[static_cast<std::size_t>(c)];
    for (int y = 0; y < cfg.size; ++y) {
      for (int x = 0; x < cfg.size; ++x) {
        // Clamp rather than assume: a source smaller than the target in one axis after rounding
        // would otherwise read out of bounds.
        const int sy = std::clamp(y0 + y, 0, rh - 1);
        const int sx = std::clamp(x0 + x, 0, rw - 1);
        const float v = resized[(static_cast<std::size_t>(sy) * rw + sx) * 3 + c] / 255.0f;
        out[static_cast<std::size_t>(c) * plane + static_cast<std::size_t>(y) * cfg.size + x] =
            (v - mean) / sd;
      }
    }
  }
  return true;
}

}  // namespace qorvix::vision
