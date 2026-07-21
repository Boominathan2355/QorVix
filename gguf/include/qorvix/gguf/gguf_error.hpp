#pragma once

#include <stdexcept>

namespace qorvix::gguf {

// Thrown for any malformed or truncated GGUF input. Part of the public parse API (GgufFile::parse
// and ::open throw it), so it lives in its own header rather than the internal reader.
class GgufParseError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace qorvix::gguf
