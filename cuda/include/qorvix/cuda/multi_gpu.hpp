#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "qorvix/cuda/backend.hpp"    // SelfTestResult
#include "qorvix/cuda/gpu_model.hpp"  // GpuModelConfig, GpuWeight

// Multi-GPU support (SPEC Phase 10): device topology discovery, a collective-communication seam,
// and the tensor-parallel sharding plan.
//
// The whole API is callable in CPU-only builds (the stub reports one degenerate "device" and a
// world size of 1), so callers never need #ifdefs — the same shape as backend.hpp.
//
// DESIGN NOTE — why the sharding math is separated from the transport:
// Tensor parallelism is two independent problems. (1) *Which* slice of each weight a rank owns and
// where the partial sums must be summed — pure integer math, no GPU involved, and where essentially
// all the real bugs live (wrong split axis, GQA head groups straddling a rank, a column split that
// lands mid-quantization-block). (2) Actually moving bytes between devices — NCCL. Splitting them
// means (1) is verifiable on a single GPU, or with no GPU at all, by simulating N ranks and
// summing locally; only (2) needs real multi-GPU hardware. See tensorParallelSelfTest().
namespace qorvix::cuda {

// ---- device topology -----------------------------------------------------------------------

// Peer-to-peer reachability between two devices. NVLink is ~10x the bandwidth of PCIe for
// all-reduce, so the planner reports it: a TP group should stay inside an NVLink island.
enum class PeerLink : std::uint8_t {
  None = 0,   // no P2P; transfers must stage through host memory
  Pcie = 1,   // P2P over PCIe
  Nvlink = 2  // P2P over NVLink/NVSwitch
};

struct DeviceTopology {
  int deviceCount = 0;
  // Row-major [deviceCount * deviceCount]; diagonal is Nvlink (a device reaches itself trivially).
  std::vector<PeerLink> peers;
  std::size_t minFreeMem = 0;    // smallest per-device free VRAM — bounds the shardable model size
  std::size_t totalFreeMem = 0;  // aggregate across devices

  PeerLink link(int a, int b) const {
    if (a < 0 || b < 0 || a >= deviceCount || b >= deviceCount) return PeerLink::None;
    return peers[static_cast<std::size_t>(a) * deviceCount + b];
  }
  // True iff every ordered pair in [0, n) can reach every other directly (no host staging).
  bool fullyConnected(int n) const;
};

// Queries devices and probes P2P reachability. Returns an empty topology (deviceCount 0) when no
// CUDA device is present or CUDA isn't built in.
DeviceTopology queryTopology();

// ---- quantization block traits -------------------------------------------------------------

// A tensor is stored as blocks of `blockSize` elements occupying `typeSize` bytes each. Kept local
// to the cuda module (mirroring cuda_backend.cu) so this module stays independent of the gguf
// types, matching the rationale in gpu_model.hpp.
struct QuantBlockTraits {
  int blockSize = 1;
  int typeSize = 4;
};
// False for a type this module can't shard (unknown / not implemented).
bool quantTraits(std::uint32_t ggmlType, QuantBlockTraits& out);

// ---- tensor-parallel plan ------------------------------------------------------------------

// A half-open slice [begin, begin+count) of one axis of one weight.
struct Slice {
  int begin = 0;
  int count = 0;
  int end() const { return begin + count; }
  bool empty() const { return count <= 0; }
};

// What one rank owns of one transformer layer under tensor parallelism.
//
// Column-parallel (the weight's *rows*, i.e. its output dim, are split): wq, wk, wv, ffnGate,
// ffnUp. Each rank produces a slice of the output and needs no communication to do so.
// Row-parallel (the weight's *cols*, i.e. its input dim, are split): wo, ffnDown. Each rank
// consumes its own slice of the activation and produces a PARTIAL sum over the full output; the
// ranks' partials must be all-reduced (summed) to form the true output.
//
// The two are deliberately paired: wo's column split is exactly the wq row split, and ffnDown's
// column split is exactly the ffnGate/ffnUp row split, so a rank's column-parallel output feeds
// straight into its row-parallel input with no reshuffle. That pairing is the invariant this plan
// exists to guarantee.
struct TensorParallelPlan {
  int worldSize = 1;
  int rank = 0;

  Slice qHeads;   // query heads owned (in heads)
  Slice kvHeads;  // key/value heads owned (in heads)

  Slice qRows;   // rows of wq owned  == qHeads scaled by headDim
  Slice kvRows;  // rows of wk / wv owned == kvHeads scaled by headDim
  Slice woCols;  // cols of wo owned  == qRows (input dim of the o-projection)

  Slice ffnRows;      // rows of ffnGate / ffnUp owned
  Slice ffnDownCols;  // cols of ffnDown owned == ffnRows

  // Per-rank sizes the sharded GpuModelConfig needs.
  int localHeads() const { return qHeads.count; }
  int localKvHeads() const { return kvHeads.count; }
  int localFfn() const { return ffnRows.count; }
};

// Builds the plan for `rank` of `worldSize` over `cfg`.
//
// Splits are NOT required to be even. The FFN is split at quantization-block granularity (the
// remainder is spread over the first ranks), because a column split that lands mid-block would
// slice a shared fp16 scale and make the shard undecodable. Real models hit this: TinyLlama's
// ffn=5632 is 22 K-quant superblocks, and 22 % 4 != 0, so an even 4-way split is NOT block
// aligned — requiring divisibility would reject TP=4 on a model that shards fine unevenly.
//
// `ffnDownType` / `woType` are the ggml types of the row-parallel weights, which set the block
// granularity of their column splits. Returns false with `err` set when the config cannot be
// sharded at this world size (see the GQA constraint below).
bool planTensorParallel(const GpuModelConfig& cfg, int worldSize, int rank,
                        std::uint32_t woType, std::uint32_t ffnDownType,
                        TensorParallelPlan& out, std::string& err);

// Largest world size `cfg` can be sharded to. Bounded by the GQA KV heads: each rank must own at
// least one WHOLE key/value head, since a KV head is the unit an attention query group reads. A
// rank holding half a KV head would have to fetch the other half every step, which defeats the
// point. For TinyLlama (nKv=4) that caps tensor parallelism at 4 regardless of GPU count.
int maxTensorParallelWorld(const GpuModelConfig& cfg);

// ---- weight shards -------------------------------------------------------------------------

// One rank's slice of a weight, still in its original quantized layout.
//
// A ROW slice is contiguous in memory (rows are independently block-encoded along cols), so it
// borrows the source bytes with zero copy. A COLUMN slice is strided — it takes a run of blocks
// out of every row — so it must be gathered into `owned`.
class WeightShard {
 public:
  const void* data() const { return owned_.empty() ? borrowed_ : owned_.data(); }
  std::size_t bytes() const { return bytes_; }
  int rows() const { return rows_; }
  int cols() const { return cols_; }
  std::uint32_t ggmlType() const { return type_; }
  bool valid() const { return data() != nullptr && rows_ > 0 && cols_ > 0; }
  // As a descriptor the GpuModel upload path accepts.
  GpuWeight asGpuWeight() const { return GpuWeight{data(), type_, rows_, cols_}; }

  friend bool shardRows(const GpuWeight&, const Slice&, WeightShard&, std::string&);
  friend bool shardCols(const GpuWeight&, const Slice&, WeightShard&, std::string&);

 private:
  std::vector<std::uint8_t> owned_;      // populated only for column slices
  const void* borrowed_ = nullptr;       // points into the source for row slices
  std::size_t bytes_ = 0;
  int rows_ = 0, cols_ = 0;
  std::uint32_t type_ = 0;
};

// Row slice (output-dim / column-parallel split). Zero-copy: borrows `w`'s bytes, which must
// outlive the shard. Any row boundary is legal.
bool shardRows(const GpuWeight& w, const Slice& rows, WeightShard& out, std::string& err);

// Column slice (input-dim / row-parallel split). Copies, because the slice is strided. `cols`
// must be block-aligned for the weight's quant type — a split inside a block would cut a shared
// scale away from the quants it scales.
bool shardCols(const GpuWeight& w, const Slice& cols, WeightShard& out, std::string& err);

// ---- collectives ---------------------------------------------------------------------------

// The communication seam tensor parallelism needs. A row-parallel matmul leaves each rank with a
// partial sum; allReduceSum turns the ranks' partials into the true result on every rank. That is
// the ONLY collective a tensor-parallel decode step requires (twice per layer: after the
// o-projection and after the FFN down-projection).
//
// Implementations: a single-rank no-op (today's path), a local simulator that sums ranks'
// buffers on one device (the verification vehicle — see tensorParallelSelfTest), and NCCL.
class ICollective {
 public:
  virtual ~ICollective() = default;
  virtual int worldSize() const = 0;
  virtual int rank() const = 0;
  // Sums `n` floats across all ranks in place, leaving the total on every rank.
  virtual bool allReduceSum(float* buf, std::size_t n) = 0;
  virtual bool barrier() = 0;
  virtual std::string backendName() const = 0;
};

// world=1 no-op: allReduceSum leaves the buffer untouched (the sum of one partial IS the total).
// Always available, including in CPU-only builds.
std::unique_ptr<ICollective> makeSingleRankCollective();

// Simulated multi-rank collective for multi-rank testing without hardware.
std::unique_ptr<ICollective> makeSimulatedCollective(int worldSize, int rank);

// True iff this binary was built against NCCL.
bool builtWithNccl() noexcept;

// ---- self-tests ----------------------------------------------------------------------------

// Verifies the sharding math WITHOUT needing multiple GPUs: builds a small synthetic layer, runs
// it whole, then re-runs it split across N simulated ranks on this one device (summing the
// row-parallel partials locally in place of an all-reduce) and compares. A mismatch means the
// split axes, the GQA head grouping, or the block alignment are wrong — which is what would
// actually break on real multi-GPU. Covers N = 2 and 4.
SelfTestResult tensorParallelSelfTest();

}  // namespace qorvix::cuda
