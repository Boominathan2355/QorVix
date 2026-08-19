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
// partial sum; the all-reduce turns the ranks' partials into the true result on every rank. That
// is the ONLY collective a tensor-parallel decode step requires (twice per layer: after the
// o-projection and after the FFN down-projection).
//
// WHY THE SEAM IS GROUP-SCOPED AND NOT ONE OBJECT PER RANK (revised in Part b, when the transport
// was actually built): Qorvix is single-process by design, so the natural driver is ONE host
// thread that launches every rank's kernels on that rank's own stream. The launches are
// asynchronous, so the ranks still execute concurrently on their devices — the only place they
// synchronize is the all-reduce itself. With a per-rank `allReduceSum(buf)`, rank 0's call would
// have to block until rank 1 had been driven, which a single thread cannot do without deadlocking;
// NCCL's own answer to exactly this is ncclGroupStart/ncclGroupEnd, i.e. reducing at GROUP
// granularity. Putting the group in the type instead of in a comment makes the deadlock
// unrepresentable, and every implementation below maps onto it directly.
class ICollectiveGroup {
 public:
  virtual ~ICollectiveGroup() = default;

  virtual int worldSize() const = 0;
  // CUDA device rank `r` runs on. -1 for the host-memory group (tests / CPU-only builds).
  // Ranks MAY share a device — that is how the sharded model is verified on a single GPU.
  virtual int deviceOf(int rank) const = 0;
  // The stream rank `r`'s kernels must be launched on, as a cudaStream_t (void* so this header
  // stays toolkit-free). The all-reduce is enqueued on these same streams, so ordering against
  // the rank's own kernels is automatic and no explicit sync is needed mid-layer. nullptr means
  // "no stream" (host group, or the degenerate world=1 group) — use the default stream.
  virtual void* stream(int rank) const = 0;

  // Sums `n` floats elementwise across all ranks. `bufs[r]` is rank r's buffer, in the memory
  // that rank lives in (device memory on deviceOf(r), or host memory when deviceOf is -1). On
  // success EVERY buffer holds the total. Taking all ranks at once is what makes the single-thread
  // driver safe: there is no half-completed state a caller could read.
  virtual bool allReduceSum(float* const* bufs, std::size_t n) = 0;

  // Waits until every rank's queued work has completed.
  virtual bool barrier() = 0;
  virtual std::string backendName() const = 0;
};

// world=1: the single partial IS the total, so allReduceSum is a no-op. Not a placeholder — it is
// the correct reduction over one rank, and it is what the unsharded path uses. Always available.
std::unique_ptr<ICollectiveGroup> makeSingleRankGroup(int device = 0);

// Host-memory group: `bufs` point at ordinary host allocations and the sum is done on the CPU.
// CUDA-free (it lives in tensor_parallel.cpp), so the seam's contract is unit-testable in a build
// with no toolkit at all — which is the same reason the sharding math lives there.
std::unique_ptr<ICollectiveGroup> makeHostCollectiveGroup(int worldSize);

// Device group staging through host memory: copies each rank's buffer D2H, sums on the host, and
// copies the total back to every rank. Two jobs, both real: it is the correct fallback when the
// devices have no P2P path between them (see PeerLink::None), and — because it does not care
// whether two ranks name the same device — it is what lets the sharded model run N ranks on ONE
// GPU, which is how tensor parallelism gets execution-verified without multi-GPU hardware.
// `devices` gives one device index per rank and may repeat. Null with `err` set on failure.
std::unique_ptr<ICollectiveGroup> makeHostStagedDeviceGroup(const std::vector<int>& devices,
                                                            std::string& err);

// NCCL group: one rank per device in a single process (ncclCommInitAll — no MPI, no network
// bootstrap). The fast path, and the only one that uses NVLink/PCIe P2P bandwidth. Requires
// DISTINCT device indices (NCCL rejects two ranks on one device) and a build with NCCL. Null with
// `err` set otherwise.
std::unique_ptr<ICollectiveGroup> makeNcclCollectiveGroup(const std::vector<int>& devices,
                                                          std::string& err);

// THE selection point — one place decides which transport a rank set gets, so no call site ever
// branches on it: NCCL when it is built in and the devices are distinct, otherwise the host-staged
// device group, and the world=1 no-op for a single rank. `err` is set only when nothing works.
std::unique_ptr<ICollectiveGroup> makeCollectiveGroup(const std::vector<int>& devices,
                                                      std::string& err);

// True iff this binary was built against NCCL.
bool builtWithNccl() noexcept;

// ---- sharded model -------------------------------------------------------------------------

// Builds a TENSOR-PARALLEL GpuModel: the plan of Part a, actually executed.
//
// The return type is the plain GpuModel interface, which is the whole point — GpuEngine, the
// scheduler, `serve` and every HTTP path already drive that seam, so multi-GPU arrives without a
// single line changing above this call. There is no "multi-GPU code path" in the runtime; there is
// one model interface with two implementations behind it.
//
// `devices` gives one CUDA device index per rank, so its size IS the world size. REPEATING an
// index is legal and deliberate: it places several ranks on one GPU (via the host-staged
// collective, which NCCL cannot do) and runs the identical sharded code, which is how the sharded
// path is execution-verified on a machine with a single GPU. On distinct devices with NCCL built
// in, the same call is the real thing.
//
// The weights are the UNSHARDED model, exactly as createGpuModel takes them; this function does
// the slicing, so callers never build per-rank weight lists. Fails with `error` set when the
// config cannot be sharded at this world size (see planTensorParallel / maxTensorParallelWorld),
// when a device is missing, or when a rank runs out of VRAM.
std::unique_ptr<GpuModel> createShardedGpuModel(const GpuModelConfig& cfg,
                                                const float* tokenEmbdF32, const float* outputNorm,
                                                const GpuWeight& output,
                                                const std::vector<GpuLayer>& layers,
                                                const std::vector<int>& devices, std::string& error,
                                                int maxSessions = 1);

// ---- self-tests ----------------------------------------------------------------------------

// Verifies the sharding math WITHOUT needing multiple GPUs: builds a small synthetic layer, runs
// it whole, then re-runs it split across N simulated ranks on this one device (summing the
// row-parallel partials locally in place of an all-reduce) and compares. A mismatch means the
// split axes, the GQA head grouping, or the block alignment are wrong — which is what would
// actually break on real multi-GPU. Covers N = 2 and 4.
SelfTestResult tensorParallelSelfTest();

// Verifies the collective TRANSPORT rather than the sharding math: gives each rank a buffer whose
// contents make the expected sum knowable in closed form, all-reduces, and checks every rank's
// buffer — the contract being that the total lands on all of them, not just one. Runs whichever
// transport makeCollectiveGroup() selected, so on a multi-GPU host with NCCL built in this is the
// NCCL path being exercised, and on a single GPU it is the host-staged one.
SelfTestResult collectiveSelfTest();

}  // namespace qorvix::cuda
