#include "qorvix/rag/vector_store.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>

#include "qorvix/runtime/pooling.hpp"

namespace qorvix::rag {

namespace {

void putU32(std::ostream& out, std::uint32_t v) {
  char b[4];
  for (int i = 0; i < 4; ++i) b[i] = static_cast<char>((v >> (8 * i)) & 0xFF);
  out.write(b, 4);
}
void putU64(std::ostream& out, std::uint64_t v) {
  char b[8];
  for (int i = 0; i < 8; ++i) b[i] = static_cast<char>((v >> (8 * i)) & 0xFF);
  out.write(b, 8);
}
void putStr(std::ostream& out, const std::string& s) {
  putU64(out, s.size());
  out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

bool getU32(std::istream& in, std::uint32_t& v) {
  char b[4];
  if (!in.read(b, 4)) return false;
  v = 0;
  for (int i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(static_cast<unsigned char>(b[i])) << (8 * i);
  return true;
}
bool getU64(std::istream& in, std::uint64_t& v) {
  char b[8];
  if (!in.read(b, 8)) return false;
  v = 0;
  for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(static_cast<unsigned char>(b[i])) << (8 * i);
  return true;
}
// Bounded read: a corrupt length field must not turn into a multi-gigabyte allocation.
bool getStr(std::istream& in, std::string& s, std::uint64_t limit) {
  std::uint64_t n = 0;
  if (!getU64(in, n) || n > limit) return false;
  s.resize(static_cast<std::size_t>(n));
  return n == 0 || static_cast<bool>(in.read(s.data(), static_cast<std::streamsize>(n)));
}

constexpr std::uint64_t kMaxStringBytes = 64ull * 1024 * 1024;

}  // namespace

bool VectorStore::add(Chunk chunk, const std::vector<float>& vec) {
  if (dim_ == 0) dim_ = static_cast<int>(vec.size());
  if (vec.empty() || static_cast<int>(vec.size()) != dim_) return false;
  chunks_.push_back(std::move(chunk));
  vectors_.insert(vectors_.end(), vec.begin(), vec.end());
  return true;
}

std::vector<SearchHit> VectorStore::searchDense(const std::vector<float>& query, int k) const {
  std::vector<SearchHit> hits;
  if (k <= 0 || chunks_.empty() || static_cast<int>(query.size()) != dim_) return hits;

  hits.reserve(chunks_.size());
  for (std::size_t i = 0; i < chunks_.size(); ++i) {
    hits.push_back({i, runtime::ops::cosineSimilarity(query.data(), vector(i), dim_)});
  }
  const std::size_t n = std::min(static_cast<std::size_t>(k), hits.size());
  // stable_sort so equal scores keep insertion order, which makes results reproducible across
  // runs and platforms rather than depending on the sort's internal pivoting.
  std::stable_sort(hits.begin(), hits.end(),
                   [](const SearchHit& a, const SearchHit& b) { return a.score > b.score; });
  hits.resize(n);
  return hits;
}

bool VectorStore::save(const std::filesystem::path& path, std::string& error) const {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    error = "cannot open '" + path.string() + "' for writing";
    return false;
  }
  putU32(out, kMagic);
  putU32(out, kVersion);
  putU32(out, static_cast<std::uint32_t>(dim_));
  putU32(out, 0);  // reserved, keeps the header 8-byte aligned
  putU64(out, chunks_.size());

  // Vectors first, contiguously: the layout a future zero-copy mmap load would want.
  out.write(reinterpret_cast<const char*>(vectors_.data()),
            static_cast<std::streamsize>(vectors_.size() * sizeof(float)));

  for (const auto& c : chunks_) {
    putStr(out, c.docId);
    putStr(out, c.source);
    putStr(out, c.text);
    putU32(out, static_cast<std::uint32_t>(c.index));
    putU64(out, c.byteStart);
    putU64(out, c.byteEnd);
    putU32(out, static_cast<std::uint32_t>(c.tokenCount));
  }
  if (!out) {
    error = "write failed for '" + path.string() + "'";
    return false;
  }
  return true;
}

std::optional<VectorStore> VectorStore::load(const std::filesystem::path& path,
                                             std::string& error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "cannot open '" + path.string() + "'";
    return std::nullopt;
  }
  std::uint32_t magic = 0, version = 0, dim = 0, reserved = 0;
  std::uint64_t count = 0;
  if (!getU32(in, magic) || !getU32(in, version) || !getU32(in, dim) || !getU32(in, reserved) ||
      !getU64(in, count)) {
    error = "'" + path.string() + "' is truncated";
    return std::nullopt;
  }
  if (magic != kMagic) {
    error = "'" + path.string() + "' is not a qorvix vector store (bad magic)";
    return std::nullopt;
  }
  if (version != kVersion) {
    error = "'" + path.string() + "' is format version " + std::to_string(version) +
            ", this build reads version " + std::to_string(kVersion);
    return std::nullopt;
  }
  if (dim == 0 || dim > 65536) {
    error = "'" + path.string() + "' declares an implausible dimension " + std::to_string(dim);
    return std::nullopt;
  }

  VectorStore store(static_cast<int>(dim));
  store.vectors_.resize(static_cast<std::size_t>(count) * dim);
  if (count > 0 && !in.read(reinterpret_cast<char*>(store.vectors_.data()),
                            static_cast<std::streamsize>(store.vectors_.size() * sizeof(float)))) {
    error = "'" + path.string() + "' is truncated in the vector block";
    return std::nullopt;
  }

  store.chunks_.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i) {
    Chunk c;
    std::uint32_t idx = 0, toks = 0;
    if (!getStr(in, c.docId, kMaxStringBytes) || !getStr(in, c.source, kMaxStringBytes) ||
        !getStr(in, c.text, kMaxStringBytes) || !getU32(in, idx) ||
        !getU64(in, c.byteStart) || !getU64(in, c.byteEnd) || !getU32(in, toks)) {
      error = "'" + path.string() + "' is truncated at chunk " + std::to_string(i);
      return std::nullopt;
    }
    c.index = static_cast<int>(idx);
    c.tokenCount = static_cast<int>(toks);
    store.chunks_.push_back(std::move(c));
  }
  return store;
}

}  // namespace qorvix::rag
