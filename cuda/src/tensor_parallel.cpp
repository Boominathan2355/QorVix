// Tensor-parallel sharding math and weight slicing (SPEC Phase 10).
//
// Deliberately free of any CUDA dependency so it compiles into BOTH the real backend and the CPU
// stub: the split logic is where the bugs are, and it can be exercised on a machine with no GPU.
// The device-side pieces (topology probing, the simulated multi-rank self-test) live in
// cuda_backend.cu / cuda_backend_stub.cpp.

#include <cstring>
#include <string>

#include "qorvix/cuda/multi_gpu.hpp"

namespace qorvix::cuda {
namespace {

// Splits `total` units into `worldSize` parts as evenly as possible, giving the first
// `total % worldSize` parts one extra unit. Returns the slice belonging to `rank`.
Slice evenSplit(int total, int worldSize, int rank) {
  const int base = total / worldSize;
  const int extra = total % worldSize;
  const int begin = rank * base + (rank < extra ? rank : extra);
  return Slice{begin, base + (rank < extra ? 1 : 0)};
}

}  // namespace

bool DeviceTopology::fullyConnected(int n) const {
  if (n <= 0 || n > deviceCount) return false;
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b)
      if (a != b && link(a, b) == PeerLink::None) return false;
  return true;
}

bool quantTraits(std::uint32_t ggmlType, QuantBlockTraits& out) {
  switch (ggmlType) {
    case 0:  out = {1, 4}; return true;      // F32
    case 1:  out = {1, 2}; return true;      // F16
    case 8:  out = {32, 34}; return true;    // Q8_0: fp16 scale + 32 int8
    case 12: out = {256, 144}; return true;  // Q4_K
    case 14: out = {256, 210}; return true;  // Q6_K
    default: return false;
  }
}

int maxTensorParallelWorld(const GpuModelConfig& cfg) {
  if (cfg.nKv <= 0 || cfg.nHeads <= 0) return 1;
  // Each rank must own >=1 whole KV head, and query heads must divide into the same rank count
  // so every query group sits with the KV head it attends to.
  int best = 1;
  for (int w = 1; w <= cfg.nKv; ++w)
    if (cfg.nKv % w == 0 && cfg.nHeads % w == 0) best = w;
  return best;
}

bool planTensorParallel(const GpuModelConfig& cfg, int worldSize, int rank,
                        std::uint32_t woType, std::uint32_t ffnDownType,
                        TensorParallelPlan& out, std::string& err) {
  if (worldSize < 1) { err = "worldSize must be >= 1"; return false; }
  if (rank < 0 || rank >= worldSize) { err = "rank out of range for worldSize"; return false; }
  if (cfg.nHeads <= 0 || cfg.nKv <= 0 || cfg.headDim <= 0 || cfg.ffn <= 0) {
    err = "invalid model config (heads/kv/headDim/ffn must be > 0)";
    return false;
  }

  // --- attention: split by whole KV heads, keeping each query group with its KV head ----------
  // GQA groups cfg.nHeads/cfg.nKv query heads onto each KV head. Splitting anywhere else would
  // put a query head on a different rank from the K/V it must attend to, forcing a per-step
  // exchange of the KV cache — so both counts must divide by worldSize.
  if (cfg.nKv % worldSize != 0) {
    err = "cannot shard " + std::to_string(cfg.nKv) + " KV heads across " +
          std::to_string(worldSize) + " ranks (each rank needs whole KV heads; max TP = " +
          std::to_string(maxTensorParallelWorld(cfg)) + ")";
    return false;
  }
  if (cfg.nHeads % worldSize != 0) {
    err = "cannot shard " + std::to_string(cfg.nHeads) + " query heads across " +
          std::to_string(worldSize) + " ranks";
    return false;
  }

  out.worldSize = worldSize;
  out.rank = rank;
  out.qHeads = evenSplit(cfg.nHeads, worldSize, rank);
  out.kvHeads = evenSplit(cfg.nKv, worldSize, rank);
  out.qRows = Slice{out.qHeads.begin * cfg.headDim, out.qHeads.count * cfg.headDim};
  out.kvRows = Slice{out.kvHeads.begin * cfg.headDim, out.kvHeads.count * cfg.headDim};
  // wo is row-parallel: its INPUT dim is exactly wq's output dim, so its column slice mirrors
  // qRows. That pairing is what lets a rank feed its own attention output straight into wo.
  out.woCols = out.qRows;

  // The wo column split must not cut a quantization block in half.
  QuantBlockTraits woT;
  if (!quantTraits(woType, woT)) {
    err = "unsupported ggml type " + std::to_string(woType) + " for wo";
    return false;
  }
  if (out.woCols.begin % woT.blockSize != 0 || out.woCols.count % woT.blockSize != 0) {
    err = "wo column split (" + std::to_string(out.woCols.begin) + "+" +
          std::to_string(out.woCols.count) + ") is not aligned to the " +
          std::to_string(woT.blockSize) + "-element quant block";
    return false;
  }

  // --- FFN: split at quantization-block granularity ------------------------------------------
  // ffnDown is row-parallel, so the split point must land on a block boundary. Splitting the
  // BLOCK count (rather than the element count) makes that automatic and tolerates a ffn size
  // that isn't divisible by worldSize — e.g. TinyLlama's 5632 = 22 superblocks over 4 ranks
  // becomes 6/6/5/5 blocks, which an even element split could never express.
  QuantBlockTraits dnT;
  if (!quantTraits(ffnDownType, dnT)) {
    err = "unsupported ggml type " + std::to_string(ffnDownType) + " for ffnDown";
    return false;
  }
  if (cfg.ffn % dnT.blockSize != 0) {
    err = "ffn size " + std::to_string(cfg.ffn) + " is not a multiple of the " +
          std::to_string(dnT.blockSize) + "-element quant block";
    return false;
  }
  const int ffnBlocks = cfg.ffn / dnT.blockSize;
  if (ffnBlocks < worldSize) {
    err = "ffn has only " + std::to_string(ffnBlocks) + " quant blocks; cannot give " +
          std::to_string(worldSize) + " ranks one each";
    return false;
  }
  const Slice blk = evenSplit(ffnBlocks, worldSize, rank);
  out.ffnRows = Slice{blk.begin * dnT.blockSize, blk.count * dnT.blockSize};
  out.ffnDownCols = out.ffnRows;  // same pairing as wo/wq
  return true;
}

// ---- weight slicing --------------------------------------------------------------------------

namespace {
// Bytes one row of `cols` elements occupies in `t`'s layout.
bool rowBytesOf(const QuantBlockTraits& t, int cols, std::size_t& out, std::string& err) {
  if (cols % t.blockSize != 0) {
    err = "cols " + std::to_string(cols) + " is not a multiple of block size " +
          std::to_string(t.blockSize);
    return false;
  }
  out = static_cast<std::size_t>(cols / t.blockSize) * t.typeSize;
  return true;
}
}  // namespace

bool shardRows(const GpuWeight& w, const Slice& rows, WeightShard& out, std::string& err) {
  if (!w.host) { err = "shardRows: null weight"; return false; }
  if (rows.begin < 0 || rows.count <= 0 || rows.end() > w.rows) {
    err = "shardRows: slice [" + std::to_string(rows.begin) + "," + std::to_string(rows.end()) +
          ") out of range for " + std::to_string(w.rows) + " rows";
    return false;
  }
  QuantBlockTraits t;
  if (!quantTraits(w.ggmlType, t)) {
    err = "shardRows: unsupported ggml type " + std::to_string(w.ggmlType);
    return false;
  }
  std::size_t rowBytes = 0;
  if (!rowBytesOf(t, w.cols, rowBytes, err)) return false;

  // Rows are independently encoded along cols, so a row range is one contiguous byte range —
  // borrow it rather than copying (these are hundreds of MB for a real model).
  out.owned_.clear();
  out.borrowed_ = static_cast<const std::uint8_t*>(w.host) +
                  static_cast<std::size_t>(rows.begin) * rowBytes;
  out.bytes_ = static_cast<std::size_t>(rows.count) * rowBytes;
  out.rows_ = rows.count;
  out.cols_ = w.cols;
  out.type_ = w.ggmlType;
  return true;
}

bool shardCols(const GpuWeight& w, const Slice& cols, WeightShard& out, std::string& err) {
  if (!w.host) { err = "shardCols: null weight"; return false; }
  if (cols.begin < 0 || cols.count <= 0 || cols.end() > w.cols) {
    err = "shardCols: slice [" + std::to_string(cols.begin) + "," + std::to_string(cols.end()) +
          ") out of range for " + std::to_string(w.cols) + " cols";
    return false;
  }
  QuantBlockTraits t;
  if (!quantTraits(w.ggmlType, t)) {
    err = "shardCols: unsupported ggml type " + std::to_string(w.ggmlType);
    return false;
  }
  if (cols.begin % t.blockSize != 0 || cols.count % t.blockSize != 0) {
    err = "shardCols: slice must be aligned to the " + std::to_string(t.blockSize) +
          "-element quant block (got begin=" + std::to_string(cols.begin) +
          ", count=" + std::to_string(cols.count) + ")";
    return false;
  }
  std::size_t srcRowBytes = 0, dstRowBytes = 0;
  if (!rowBytesOf(t, w.cols, srcRowBytes, err)) return false;
  if (!rowBytesOf(t, cols.count, dstRowBytes, err)) return false;
  const std::size_t byteOff = static_cast<std::size_t>(cols.begin / t.blockSize) * t.typeSize;

  // A column range takes a run of blocks out of every row — strided, so it must be gathered.
  out.owned_.assign(static_cast<std::size_t>(w.rows) * dstRowBytes, 0);
  const auto* src = static_cast<const std::uint8_t*>(w.host);
  for (int r = 0; r < w.rows; ++r)
    std::memcpy(out.owned_.data() + static_cast<std::size_t>(r) * dstRowBytes,
                src + static_cast<std::size_t>(r) * srcRowBytes + byteOff, dstRowBytes);
  out.borrowed_ = nullptr;
  out.bytes_ = out.owned_.size();
  out.rows_ = w.rows;
  out.cols_ = cols.count;
  out.type_ = w.ggmlType;
  return true;
}

// ---- single-rank collective ------------------------------------------------------------------

namespace {
// world=1: every "partial" is already the total, so allReduceSum is a no-op. Real, not a
// placeholder — it is the correct implementation of summing across one rank.
class SingleRankCollective final : public ICollective {
 public:
  int worldSize() const override { return 1; }
  int rank() const override { return 0; }
  bool allReduceSum(float*, std::size_t) override { return true; }
  bool barrier() override { return true; }
  std::string backendName() const override { return "single-rank"; }
};

class SimulatedMultiRankCollective final : public ICollective {
 public:
  SimulatedMultiRankCollective(int worldSize, int rank) : worldSize_(worldSize), rank_(rank) {}
  int worldSize() const override { return worldSize_; }
  int rank() const override { return rank_; }
  bool allReduceSum(float*, std::size_t) override { return true; }
  bool barrier() override { return true; }
  std::string backendName() const override { return "simulated-multirank"; }
 private:
  int worldSize_ = 1;
  int rank_ = 0;
};
}  // namespace

std::unique_ptr<ICollective> makeSingleRankCollective() {
  return std::make_unique<SingleRankCollective>();
}

std::unique_ptr<ICollective> makeSimulatedCollective(int worldSize, int rank) {
  return std::make_unique<SimulatedMultiRankCollective>(worldSize, rank);
}

#ifndef QORVIX_WITH_NCCL
bool builtWithNccl() noexcept { return false; }
#endif

}  // namespace qorvix::cuda
