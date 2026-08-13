#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// DEFLATE (RFC 1951) and zlib (RFC 1950) decompression, from scratch.
//
// Lives here because PNG needs it, but there is nothing image-specific about it: the same decoder
// is the honest prerequisite for the PDF (FlateDecode) and DOCX (ZIP) loaders that rag/loaders.hpp
// currently refuses. Promote it to its own module when that second consumer lands — the same rule
// applied to IVectorStore: add the seam when there is a second thing behind it, not before.
//
// The format is small and exactly specified, which is why it is worth writing rather than taking a
// dependency: fixed and dynamic Huffman blocks, stored blocks, and a 32 KiB back-reference window.
namespace qorvix::vision {

// Raw DEFLATE stream (no zlib header). Returns false with `error` set on malformed input.
//
// `sizeHint` pre-reserves the output; it is a hint only, never a limit. `maxOutput` IS a limit and
// exists because a crafted stream can expand ~1000x — a decompression bomb must fail loudly rather
// than exhaust memory.
bool inflateRaw(const std::uint8_t* data, std::size_t size, std::vector<std::uint8_t>& out,
                std::string& error, std::size_t sizeHint = 0,
                std::size_t maxOutput = 512ull * 1024 * 1024);

// zlib wrapper (RFC 1950): 2-byte header, DEFLATE payload, Adler-32 trailer. The checksum IS
// verified — a silently corrupt image is worse than a rejected one, since it becomes a plausible
// wrong embedding rather than an error.
bool inflateZlib(const std::uint8_t* data, std::size_t size, std::vector<std::uint8_t>& out,
                 std::string& error, std::size_t sizeHint = 0,
                 std::size_t maxOutput = 512ull * 1024 * 1024);

// Adler-32 as specified by RFC 1950, exposed for tests.
std::uint32_t adler32(const std::uint8_t* data, std::size_t size);

}  // namespace qorvix::vision
