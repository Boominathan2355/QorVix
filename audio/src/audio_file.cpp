#include "qorvix/audio/audio_file.hpp"

#include <cstring>
#include <fstream>

namespace qorvix::audio {
namespace {

// WAVE format tags. EXTENSIBLE is not a format itself: it says "the real tag is the first two
// bytes of the subformat GUID", which is how 24-bit and multichannel files are usually written.
constexpr std::uint16_t kFormatPcm = 0x0001;
constexpr std::uint16_t kFormatFloat = 0x0003;
constexpr std::uint16_t kFormatExtensible = 0xFFFE;

std::uint16_t rd16(const std::uint8_t* p) {
  return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}
std::uint32_t rd32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}
bool tagIs(const std::uint8_t* p, const char* fourcc) {
  return std::memcmp(p, fourcc, 4) == 0;
}

// Names the container so a user knows this is an unimplemented format rather than a corrupt file —
// the same courtesy vision::decodeImage extends to JPEG.
const char* compressedFormatName(const std::uint8_t* d, std::size_t n) {
  if (n >= 4 && tagIs(d, "fLaC")) return "FLAC";
  if (n >= 4 && tagIs(d, "OggS")) return "OGG";
  if (n >= 12 && tagIs(d, "RIFF") && !tagIs(d + 8, "WAVE")) return "a non-WAVE RIFF container";
  if (n >= 12 && tagIs(d + 4, "ftyp")) return "MP4/M4A";
  if (n >= 3 && d[0] == 'I' && d[1] == 'D' && d[2] == '3') return "MP3";
  // MPEG audio frame sync: 11 set bits.
  if (n >= 2 && d[0] == 0xFF && (d[1] & 0xE0) == 0xE0) return "MP3";
  return nullptr;
}

// One sample -> float in [-1, 1]. Integer PCM is signed and little-endian except at 8 bits, where
// WAV stores it unsigned with a 128 bias; getting that backwards produces audio that is silent-ish
// and offset rather than obviously broken, so it is handled explicitly.
float sampleToFloat(const std::uint8_t* p, std::uint16_t bits, bool isFloat) {
  if (isFloat) {
    if (bits == 32) {
      float v = 0.0f;
      std::memcpy(&v, p, 4);
      return v;
    }
    double v = 0.0;
    std::memcpy(&v, p, 8);
    return static_cast<float>(v);
  }
  switch (bits) {
    case 8:
      return (static_cast<int>(p[0]) - 128) / 128.0f;
    case 16: {
      const auto v = static_cast<std::int16_t>(rd16(p));
      return v / 32768.0f;
    }
    case 24: {
      std::int32_t v = static_cast<std::int32_t>(p[0]) | (static_cast<std::int32_t>(p[1]) << 8) |
                       (static_cast<std::int32_t>(p[2]) << 16);
      if (v & 0x00800000) v |= ~0x00FFFFFF;  // sign-extend the 24-bit value
      return static_cast<float>(v) / 8388608.0f;
    }
    case 32: {
      const auto v = static_cast<std::int32_t>(rd32(p));
      return static_cast<float>(v) / 2147483648.0f;
    }
    default:
      return 0.0f;
  }
}

}  // namespace

bool decodeAudio(const std::uint8_t* data, std::size_t size, AudioBuffer& out,
                 std::string& error) {
  if (!data || size < 12) {
    error = "not an audio file (too short)";
    return false;
  }
  if (!tagIs(data, "RIFF") || !tagIs(data + 8, "WAVE")) {
    if (const char* name = compressedFormatName(data, size)) {
      error = std::string(name) +
              " is not supported — qorvix reads uncompressed WAV only. Convert with: "
              "ffmpeg -i <in> -ac 1 -ar 16000 -c:a pcm_s16le out.wav";
      return false;
    }
    error = "not a RIFF/WAVE file";
    return false;
  }

  std::uint16_t format = 0, channels = 0, bits = 0;
  std::uint32_t rate = 0;
  bool haveFmt = false;
  const std::uint8_t* pcm = nullptr;
  std::size_t pcmBytes = 0;

  // Chunk walk. Unknown chunks (LIST, fact, cue, and the mountain of metadata editors add) are
  // skipped rather than treated as an error, which is the whole point of a chunked container.
  std::size_t off = 12;
  while (off + 8 <= size) {
    const std::uint8_t* id = data + off;
    const std::uint32_t len = rd32(data + off + 4);
    const std::size_t body = off + 8;
    if (body + len > size) {
      // A truncated final chunk is common in streamed captures; take what is there for `data`,
      // but a truncated `fmt ` means nothing can be decoded.
      if (tagIs(id, "data") && body < size) {
        pcm = data + body;
        pcmBytes = size - body;
        break;
      }
      error = "truncated WAV chunk";
      return false;
    }
    if (tagIs(id, "fmt ") && len >= 16) {
      format = rd16(data + body);
      channels = rd16(data + body + 2);
      rate = rd32(data + body + 4);
      bits = rd16(data + body + 14);
      if (format == kFormatExtensible && len >= 26) format = rd16(data + body + 24);
      haveFmt = true;
    } else if (tagIs(id, "data")) {
      pcm = data + body;
      pcmBytes = len;
    }
    off = body + len + (len & 1);  // chunks are word-aligned; odd lengths carry a pad byte
  }

  if (!haveFmt) {
    error = "WAV has no fmt chunk";
    return false;
  }
  if (!pcm || pcmBytes == 0) {
    error = "WAV has no audio data";
    return false;
  }
  const bool isFloat = format == kFormatFloat;
  if (format != kFormatPcm && !isFloat) {
    error = "WAV codec " + std::to_string(format) +
            " is compressed or unsupported — qorvix reads PCM and IEEE float only";
    return false;
  }
  if (channels == 0 || rate == 0) {
    error = "WAV header declares zero channels or a zero sample rate";
    return false;
  }
  const int bytesPerSample = bits / 8;
  if (bytesPerSample <= 0 || (isFloat ? (bits != 32 && bits != 64)
                                      : (bits != 8 && bits != 16 && bits != 24 && bits != 32))) {
    error = "unsupported WAV bit depth " + std::to_string(bits);
    return false;
  }

  const std::size_t frameBytes = static_cast<std::size_t>(bytesPerSample) * channels;
  const std::size_t frames = pcmBytes / frameBytes;
  if (frames == 0) {
    error = "WAV data chunk holds no complete frames";
    return false;
  }

  out.sampleRate = static_cast<int>(rate);
  out.samples.resize(frames);
  for (std::size_t f = 0; f < frames; ++f) {
    const std::uint8_t* p = pcm + f * frameBytes;
    double acc = 0.0;
    for (std::uint16_t c = 0; c < channels; ++c)
      acc += sampleToFloat(p + static_cast<std::size_t>(c) * bytesPerSample, bits, isFloat);
    out.samples[f] = static_cast<float>(acc / channels);
  }
  return true;
}

bool loadAudio(const std::filesystem::path& path, AudioBuffer& out, std::string& error) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    error = "cannot open " + path.string();
    return false;
  }
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                                  std::istreambuf_iterator<char>());
  if (bytes.empty()) {
    error = path.string() + " is empty";
    return false;
  }
  return decodeAudio(bytes.data(), bytes.size(), out, error);
}

}  // namespace qorvix::audio
