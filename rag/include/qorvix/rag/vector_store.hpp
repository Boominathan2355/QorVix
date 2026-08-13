#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "qorvix/rag/document.hpp"

namespace qorvix::rag {

// Flat vector store with exact cosine search and a self-describing on-disk format.
//
// Brute force, deliberately. 100k chunks x 384 dims is 154 MB and ~40 MFLOP per query — under
// 10 ms with the runtime's SIMD dot product. HNSW earns its build time, memory overhead and
// recall@k caveats above roughly a million chunks; below that it is complexity that trades exact
// results for nothing a user can see. The seam to add one later is searchDense().
class VectorStore {
 public:
  explicit VectorStore(int dim = 0) : dim_(dim) {}

  int dim() const noexcept { return dim_; }
  std::size_t size() const noexcept { return chunks_.size(); }
  bool empty() const noexcept { return chunks_.empty(); }

  // Appends a chunk and its embedding. False if the vector's length disagrees with dim(); the
  // first add() on a default-constructed store fixes the dimension.
  bool add(Chunk chunk, const std::vector<float>& vec);

  // Exact cosine top-k, best first. Ties keep insertion order. Returns fewer than k when the
  // store holds fewer.
  std::vector<SearchHit> searchDense(const std::vector<float>& query, int k) const;

  const Chunk& chunk(std::size_t i) const { return chunks_.at(i); }
  const std::vector<Chunk>& chunks() const noexcept { return chunks_; }
  // Row i of the [size(), dim()] embedding matrix.
  const float* vector(std::size_t i) const { return vectors_.data() + i * static_cast<std::size_t>(dim_); }

  // Native .qvx format, shaped like GGUF: magic, version, then a header the reader can validate
  // before trusting any offset. A bad magic or an unknown version is rejected rather than
  // reinterpreted — silently reading a stale layout would return plausible nonsense.
  bool save(const std::filesystem::path& path, std::string& error) const;
  static std::optional<VectorStore> load(const std::filesystem::path& path, std::string& error);

  static constexpr std::uint32_t kMagic = 0x58565153;  // "QVX" + version byte, little-endian
  static constexpr std::uint32_t kVersion = 1;

 private:
  int dim_ = 0;
  std::vector<Chunk> chunks_;
  std::vector<float> vectors_;  // [size(), dim()] row-major, contiguous for a linear scan
};

}  // namespace qorvix::rag
