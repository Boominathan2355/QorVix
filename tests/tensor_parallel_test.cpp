#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "qorvix/cuda/multi_gpu.hpp"

using namespace qorvix;
using namespace qorvix::cuda;

// Phase 10 tensor-parallel sharding math. These run in EVERY build, including CPU-only ones with
// no CUDA toolkit — the split logic is deliberately GPU-free (see tensor_parallel.cpp) precisely
// so it can be tested here rather than only on multi-GPU hardware.

namespace {

constexpr int kQKK = 256, kQ4KBytes = 144, kQ6KBytes = 210;

// A TinyLlama-shaped config: the real model this project validates against.
GpuModelConfig tinyllamaShape() {
  GpuModelConfig c;
  c.nLayers = 22;
  c.dModel = 2048;
  c.nHeads = 32;
  c.headDim = 64;
  c.nKv = 4;
  c.ffn = 5632;  // = 22 K-quant super-blocks — deliberately NOT divisible by 4
  c.vocab = 32000;
  c.maxSeq = 2048;
  return c;
}

// A Q4_K weight whose bytes encode (row, super-block) so slicing can be checked positionally.
std::vector<std::uint8_t> makeQ4KWeight(int rows, int nSB) {
  std::vector<std::uint8_t> w(static_cast<std::size_t>(rows) * nSB * kQ4KBytes);
  for (int r = 0; r < rows; ++r)
    for (int sb = 0; sb < nSB; ++sb) {
      std::uint8_t* blk = w.data() + (static_cast<std::size_t>(r) * nSB + sb) * kQ4KBytes;
      for (int i = 0; i < kQ4KBytes; ++i)
        blk[i] = static_cast<std::uint8_t>((r * 31 + sb * 7 + i) & 0xFF);
    }
  return w;
}

}  // namespace

TEST_CASE("quant block traits cover the shardable types", "[tp]") {
  QuantBlockTraits t;
  REQUIRE(quantTraits(0, t));   // F32
  REQUIRE(t.blockSize == 1);
  REQUIRE(t.typeSize == 4);
  REQUIRE(quantTraits(8, t));   // Q8_0
  REQUIRE(t.blockSize == 32);
  REQUIRE(t.typeSize == 34);
  REQUIRE(quantTraits(12, t));  // Q4_K
  REQUIRE(t.blockSize == 256);
  REQUIRE(t.typeSize == kQ4KBytes);
  REQUIRE(quantTraits(14, t));  // Q6_K
  REQUIRE(t.typeSize == kQ6KBytes);
  REQUIRE_FALSE(quantTraits(9999, t));  // unknown type is rejected, not guessed
}

TEST_CASE("max tensor-parallel world is bounded by the KV heads", "[tp]") {
  const auto cfg = tinyllamaShape();
  // 4 KV heads means at most 4 ranks, however many GPUs are present: a rank must own whole KV
  // heads or it would have to fetch the rest of one every decode step.
  REQUIRE(maxTensorParallelWorld(cfg) == 4);

  GpuModelConfig mha = cfg;  // no GQA: nKv == nHeads
  mha.nKv = 32;
  REQUIRE(maxTensorParallelWorld(mha) == 32);
}

TEST_CASE("plan tiles every tensor with no gaps or overlap", "[tp]") {
  const auto cfg = tinyllamaShape();
  for (int world : {1, 2, 4}) {
    int ffn = 0, q = 0, kv = 0, ffnEnd = 0, qEnd = 0, kvEnd = 0;
    for (int r = 0; r < world; ++r) {
      TensorParallelPlan p;
      std::string err;
      REQUIRE(planTensorParallel(cfg, world, r, 12u, 12u, p, err));

      // Contiguity: each rank's slice starts exactly where the previous one ended.
      REQUIRE(p.ffnRows.begin == ffnEnd);
      REQUIRE(p.qRows.begin == qEnd);
      REQUIRE(p.kvHeads.begin == kvEnd);
      ffnEnd = p.ffnRows.end();
      qEnd = p.qRows.end();
      kvEnd = p.kvHeads.end();

      ffn += p.ffnRows.count;
      q += p.qHeads.count;
      kv += p.kvHeads.count;
    }
    REQUIRE(ffn == cfg.ffn);
    REQUIRE(q == cfg.nHeads);
    REQUIRE(kv == cfg.nKv);
  }
}

TEST_CASE("row-parallel weights pair with their column-parallel producers", "[tp]") {
  // The invariant that makes TP work without a reshuffle between the two matmuls: wo consumes
  // exactly the attention output wq produced on the SAME rank, and ffnDown consumes exactly what
  // ffnGate/ffnUp produced there. If these ever drift apart the ranks silently mix activations.
  const auto cfg = tinyllamaShape();
  for (int world : {1, 2, 4})
    for (int r = 0; r < world; ++r) {
      TensorParallelPlan p;
      std::string err;
      REQUIRE(planTensorParallel(cfg, world, r, 12u, 12u, p, err));
      REQUIRE(p.woCols.begin == p.qRows.begin);
      REQUIRE(p.woCols.count == p.qRows.count);
      REQUIRE(p.ffnDownCols.begin == p.ffnRows.begin);
      REQUIRE(p.ffnDownCols.count == p.ffnRows.count);
      // Both row-parallel splits must land on quant-block boundaries.
      REQUIRE(p.woCols.begin % kQKK == 0);
      REQUIRE(p.woCols.count % kQKK == 0);
      REQUIRE(p.ffnRows.begin % kQKK == 0);
      REQUIRE(p.ffnRows.count % kQKK == 0);
    }
}

TEST_CASE("uneven FFN split keeps blocks whole when the count does not divide", "[tp]") {
  // ffn=5632 is 22 super-blocks and 22 % 4 != 0. An even ELEMENT split would put a boundary at
  // 1408, which is mid-block (1408 % 256 == 128) and would slice a shared fp16 scale away from
  // the quants it scales. Splitting the BLOCK count instead yields 6/6/5/5 and stays decodable.
  const auto cfg = tinyllamaShape();
  std::vector<int> blocks;
  for (int r = 0; r < 4; ++r) {
    TensorParallelPlan p;
    std::string err;
    REQUIRE(planTensorParallel(cfg, 4, r, 12u, 12u, p, err));
    REQUIRE(p.ffnRows.count % kQKK == 0);
    blocks.push_back(p.ffnRows.count / kQKK);
  }
  REQUIRE(blocks == std::vector<int>{6, 6, 5, 5});
}

TEST_CASE("unshardable configurations are rejected with a reason", "[tp]") {
  const auto cfg = tinyllamaShape();
  TensorParallelPlan p;
  std::string err;

  REQUIRE_FALSE(planTensorParallel(cfg, 8, 0, 12u, 12u, p, err));  // 8 > 4 KV heads
  REQUIRE_FALSE(err.empty());
  REQUIRE_FALSE(planTensorParallel(cfg, 3, 0, 12u, 12u, p, err));  // 4 % 3 != 0
  REQUIRE_FALSE(planTensorParallel(cfg, 0, 0, 12u, 12u, p, err));  // bad world size
  REQUIRE_FALSE(planTensorParallel(cfg, 2, 2, 12u, 12u, p, err));  // rank out of range
  REQUIRE_FALSE(planTensorParallel(cfg, 2, 0, 9999u, 12u, p, err));  // unknown wo type
}

TEST_CASE("row shards are contiguous and borrow the source bytes", "[tp]") {
  const int rows = 64, nSB = 4;
  auto w = makeQ4KWeight(rows, nSB);
  const GpuWeight full{w.data(), 12u, rows, nSB * kQKK};

  WeightShard sh;
  std::string err;
  REQUIRE(shardRows(full, Slice{16, 32}, sh, err));
  REQUIRE(sh.rows() == 32);
  REQUIRE(sh.cols() == nSB * kQKK);
  REQUIRE(sh.ggmlType() == 12u);
  REQUIRE(sh.bytes() == static_cast<std::size_t>(32) * nSB * kQ4KBytes);
  // Zero-copy: a row range is one contiguous byte range in the source.
  REQUIRE(sh.data() == w.data() + static_cast<std::size_t>(16) * nSB * kQ4KBytes);
  REQUIRE(std::memcmp(sh.data(), w.data() + static_cast<std::size_t>(16) * nSB * kQ4KBytes,
                      sh.bytes()) == 0);

  REQUIRE_FALSE(shardRows(full, Slice{0, rows + 1}, sh, err));  // out of range
  REQUIRE_FALSE(shardRows(full, Slice{0, 0}, sh, err));         // empty
}

TEST_CASE("column shards gather the right blocks out of every row", "[tp]") {
  const int rows = 8, nSB = 4;
  auto w = makeQ4KWeight(rows, nSB);
  const GpuWeight full{w.data(), 12u, rows, nSB * kQKK};

  // Take super-blocks [2,4) of every row.
  WeightShard sh;
  std::string err;
  REQUIRE(shardCols(full, Slice{2 * kQKK, 2 * kQKK}, sh, err));
  REQUIRE(sh.rows() == rows);
  REQUIRE(sh.cols() == 2 * kQKK);
  REQUIRE(sh.bytes() == static_cast<std::size_t>(rows) * 2 * kQ4KBytes);

  // Every gathered byte must equal the source byte it came from — this is the check that would
  // catch an off-by-one in the stride arithmetic.
  const auto* got = static_cast<const std::uint8_t*>(sh.data());
  for (int r = 0; r < rows; ++r)
    for (int sb = 0; sb < 2; ++sb)
      for (int i = 0; i < kQ4KBytes; ++i)
        REQUIRE(got[(static_cast<std::size_t>(r) * 2 + sb) * kQ4KBytes + i] ==
                w[(static_cast<std::size_t>(r) * nSB + (sb + 2)) * kQ4KBytes + i]);
}

TEST_CASE("column shards refuse to cut a quantization block in half", "[tp]") {
  const int rows = 8, nSB = 4;
  auto w = makeQ4KWeight(rows, nSB);
  const GpuWeight full{w.data(), 12u, rows, nSB * kQKK};

  WeightShard sh;
  std::string err;
  REQUIRE_FALSE(shardCols(full, Slice{128, kQKK}, sh, err));      // begin mid-block
  REQUIRE_FALSE(shardCols(full, Slice{0, 128}, sh, err));         // count mid-block
  REQUIRE_FALSE(shardCols(full, Slice{0, nSB * kQKK + 1}, sh, err));  // past the end
  REQUIRE(shardCols(full, Slice{0, kQKK}, sh, err));              // aligned is fine
}

TEST_CASE("single-rank collective is the identity", "[tp]") {
  auto c = makeSingleRankCollective();
  REQUIRE(c != nullptr);
  REQUIRE(c->worldSize() == 1);
  REQUIRE(c->rank() == 0);
  REQUIRE(c->barrier());
  // With one rank the partial IS the total, so the buffer must come back untouched.
  std::vector<float> buf{1.5f, -2.0f, 3.25f};
  const auto before = buf;
  REQUIRE(c->allReduceSum(buf.data(), buf.size()));
  REQUIRE(buf == before);
}

TEST_CASE("topology is callable and self-consistent in any build", "[tp]") {
  const auto topo = queryTopology();
  REQUIRE(topo.deviceCount >= 0);
  REQUIRE(topo.peers.size() ==
          static_cast<std::size_t>(topo.deviceCount) * topo.deviceCount);
  // Out-of-range queries report None rather than reading past the matrix.
  REQUIRE(topo.link(-1, 0) == PeerLink::None);
  REQUIRE(topo.link(0, topo.deviceCount) == PeerLink::None);
  for (int i = 0; i < topo.deviceCount; ++i)
    REQUIRE(topo.link(i, i) == PeerLink::Nvlink);  // a device always reaches itself
  REQUIRE_FALSE(topo.fullyConnected(0));
  if (!builtWithCuda()) REQUIRE(topo.deviceCount == 0);
}

TEST_CASE("tensor-parallel self-test passes on a device or reports skipped", "[tp]") {
  const auto tp = tensorParallelSelfTest();
  if (deviceCount() > 0 && builtWithCuda()) {
    REQUIRE(tp.ran);
    REQUIRE(tp.passed);
  } else {
    REQUIRE_FALSE(tp.ran);  // skipped, not failed
  }
}
