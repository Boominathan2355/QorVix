#include "qorvix/image/blocks.hpp"

namespace qorvix::image {

bool convForward(FeatureMap& out, const FeatureMap& in, const MatWeights& w, int kernel, int stride,
                 int pad, std::vector<float>& im2col, std::string& error) {
  if (kernel == 1 && stride == 1 && pad == 0) return pointwise(out, in, w.w, w.b, error);
  return conv2d(out, in, w.w, w.b, kernel, stride, pad, im2col, error);
}

bool resnetForward(const ResnetWeights& r, const FeatureMap& in, const std::vector<float>* timeEmb,
                   int groups, float eps, FeatureMap& out, BlockScratch& s, std::string& error) {
  s.in = in;
  groupNorm(s.in, r.norm1.w, r.norm1.b, groups, eps);
  siluInPlace(s.in.data.data(), s.in.size());
  if (!convForward(s.tmp, s.in, r.conv1, 3, 1, 1, s.im2col, error)) return false;

  if (timeEmb && r.timeEmb.valid()) {
    s.tembSilu = *timeEmb;
    siluInPlace(s.tembSilu.data(), s.tembSilu.size());
    s.tproj.resize(static_cast<std::size_t>(r.timeEmb.w.rows));
    runtime::wmatmulBias(s.tproj.data(), r.timeEmb.w, s.tembSilu.data(), r.timeEmb.b);
    addPerChannel(s.tmp, s.tproj.data());
  }

  groupNorm(s.tmp, r.norm2.w, r.norm2.b, groups, eps);
  siluInPlace(s.tmp.data.data(), s.tmp.size());
  if (!convForward(out, s.tmp, r.conv2, 3, 1, 1, s.im2col, error)) return false;

  // The shortcut is a 1x1 projection exactly when the channel count changed. When it did not, the
  // residual is the input itself — and an input whose shape does not match is a loader bug that
  // would otherwise read past the end of a buffer.
  if (r.skip.valid()) {
    if (!pointwise(s.skip, in, r.skip.w, r.skip.b, error)) return false;
    addInPlace(out.data.data(), s.skip.data.data(), out.size());
  } else {
    if (!out.matches(in)) {
      error = "residual block has no shortcut but goes from " + std::to_string(in.c) + " to " +
              std::to_string(out.c) + " channels";
      return false;
    }
    addInPlace(out.data.data(), in.data.data(), out.size());
  }
  return true;
}

}  // namespace qorvix::image
