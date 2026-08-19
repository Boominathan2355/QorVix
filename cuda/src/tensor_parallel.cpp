// Tensor-parallel sharding math and weight slicing (SPEC Phase 10).
//
// Deliberately free of any CUDA dependency so it compiles into BOTH the real backend and the CPU
// stub: the split logic is where the bugs are, and it can be exercised on a machine with no GPU.
// The device-side pieces (topology probing, the host-staged and NCCL collective groups, and the
// self-tests that need a device) live in cuda_backend.cu / cuda_backend_stub.cpp.

#include <cstring>
#include <string>
#include <vector>

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

// ---- toolkit-free collective groups ------------------------------------------------------------
//
// Two of the four transports need no CUDA at all, so they live here with the sharding math and are
// exercised by the unit tests in every build: the degenerate world=1 group, and a host-memory group
// that pins down what "all-reduce" means before any device is involved. The device transports
// (host-staged and NCCL) are in cuda_backend.cu / cuda_backend_stub.cpp.

namespace {

// world=1: every "partial" is already the total, so allReduceSum is a no-op. Real, not a
// placeholder — it is the correct implementation of summing across one rank, and it is what the
// unsharded path uses.
class SingleRankGroup final : public ICollectiveGroup {
 public:
  explicit SingleRankGroup(int device) : device_(device) {}
  int worldSize() const override { return 1; }
  int deviceOf(int) const override { return device_; }
  void* stream(int) const override { return nullptr; }
  bool allReduceSum(float* const*, std::size_t) override { return true; }
  bool barrier() override { return true; }
  std::string backendName() const override { return "single-rank"; }

 private:
  int device_ = 0;
};

// Host-memory group. The reference definition of the contract: sum the ranks elementwise, then
// write the total back to EVERY rank — a reduce that left the answer only on rank 0 would pass a
// naive test and then produce garbage on ranks 1..N-1 in the model, so the write-back is the part
// worth pinning.
class HostCollectiveGroup final : public ICollectiveGroup {
 public:
  explicit HostCollectiveGroup(int worldSize) : worldSize_(worldSize) {}
  int worldSize() const override { return worldSize_; }
  int deviceOf(int) const override { return -1; }  // host memory, no device
  void* stream(int) const override { return nullptr; }

  bool allReduceSum(float* const* bufs, std::size_t n) override {
    if (!bufs || n == 0) return false;
    for (int r = 0; r < worldSize_; ++r)
      if (!bufs[r]) return false;
    acc_.assign(n, 0.0f);
    for (int r = 0; r < worldSize_; ++r)
      for (std::size_t i = 0; i < n; ++i) acc_[i] += bufs[r][i];
    for (int r = 0; r < worldSize_; ++r)
      std::memcpy(bufs[r], acc_.data(), n * sizeof(float));
    return true;
  }

  bool barrier() override { return true; }
  std::string backendName() const override { return "host"; }

 private:
  int worldSize_ = 1;
  std::vector<float> acc_;  // reused across calls; the decode loop runs this twice per layer
};

}  // namespace

std::unique_ptr<ICollectiveGroup> makeSingleRankGroup(int device) {
  return std::make_unique<SingleRankGroup>(device);
}

std::unique_ptr<ICollectiveGroup> makeHostCollectiveGroup(int worldSize) {
  if (worldSize < 1) return nullptr;
  if (worldSize == 1) return std::make_unique<SingleRankGroup>(-1);
  return std::make_unique<HostCollectiveGroup>(worldSize);
}

// THE transport selection point. It lives here, in the CUDA-free file, so there is exactly ONE
// policy shared by the real and the stub build rather than one per configuration — the same
// reason the sharding math is here. Every caller asks for "a group over these ranks" and never
// learns which transport it got except through backendName().
std::unique_ptr<ICollectiveGroup> makeCollectiveGroup(const std::vector<int>& devices,
                                                      std::string& err) {
  const int world = static_cast<int>(devices.size());
  if (world < 1) {
    err = "makeCollectiveGroup: no devices given (need one device index per rank)";
    return nullptr;
  }
  if (world == 1) return makeSingleRankGroup(devices[0]);

  // NCCL cannot place two ranks on one device, and two ranks on one device is precisely the
  // single-GPU verification configuration — so the distinctness check is a routing decision, not
  // an error. A repeated index means "simulate this world on fewer GPUs", which host-staged does.
  bool distinct = true;
  for (int i = 0; i < world && distinct; ++i)
    for (int j = i + 1; j < world; ++j)
      if (devices[i] == devices[j]) { distinct = false; break; }

  std::string why;
  if (distinct) {
    std::string nerr;
    if (auto g = makeNcclCollectiveGroup(devices, nerr)) return g;
    why = "NCCL unavailable (" + nerr + "); fell back to host staging: ";
  }
  std::string herr;
  if (auto g = makeHostStagedDeviceGroup(devices, herr)) return g;
  err = why + herr;
  return nullptr;
}

#ifndef QORVIX_WITH_NCCL
bool builtWithNccl() noexcept { return false; }

std::unique_ptr<ICollectiveGroup> makeNcclCollectiveGroup(const std::vector<int>&,
                                                          std::string& err) {
  err = "NCCL not built in (rebuild with -DQORVIX_ENABLE_NCCL=ON on a host with the NCCL library)";
  return nullptr;
}
#endif

}  // namespace qorvix::cuda
