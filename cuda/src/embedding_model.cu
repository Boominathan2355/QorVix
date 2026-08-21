// CUDA BERT-family embedding engine (Phase 11c) - GPU implementation of IEmbeddingEngine.
// Mirrors CPU BertModel: post-norm LayerNorm with bias, bidirectional attention, GELU FFN,
// learned position embeddings, CLS/mean/last pooling + L2 normalize.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "qorvix/cuda/embedding_model.hpp"

namespace qorvix::cuda {
namespace {

constexpr int kEmbThreads = 256;
constexpr int kWarpBlock = 8;
constexpr int kQKK = 256;
constexpr int kQ4KB = 144;
constexpr int kQ6KB = 210;

std::size_t embWBytes(std::uint32_t t, int r, int c) {
  std::size_t n = static_cast<std::size_t>(r) * c;
  switch (t) {
    case 0: return n * 4;
    case 8: return n / 32 * 34;
    case 12: return n / kQKK * kQ4KB;
    case 14: return n / kQKK * kQ6KB;
    default: return 0;
  }
}

struct EmbW { void* d = nullptr; std::uint32_t type = 0; int rows = 0; int cols = 0; };

// ---- quantized GEMV (self-contained copies of cuda_backend.cu kernels) -----------------------

__global__ void qgemvQ8(float* out, const unsigned char* W, const float* x, int rows, int cols) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int row = blockIdx.x * kWarpBlock + warp;
  if (row >= rows) return;
  int nB = cols / 32;
  const unsigned char* rp = W + static_cast<std::size_t>(row) * nB * 34;
  float sum = 0.0f;
  for (int b = 0; b < nB; ++b) {
    const unsigned char* blk = rp + b * 34;
    float d = 0.0f;
    if (lane == 0) { unsigned short h = blk[0]|(unsigned short(blk[1])<<8); d = __half2float(__ushort_as_half(h)); }
    d = __shfl_sync(0xffffffffu, d, 0);
    sum += d * float((signed char)blk[2+lane]) * x[b*32+lane];
  }
  for (int o=16; o>0; o>>=1) sum += __shfl_down_sync(0xffffffffu, sum, o);
  if (lane==0) out[row] = sum;
}

__global__ void qgemvQ4K(float* out, const unsigned char* W, const float* x, int rows, int cols) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int row = blockIdx.x * kWarpBlock + warp;
  if (row >= rows) return;
  int nSB = cols / kQKK;
  const unsigned char* rp = W + static_cast<std::size_t>(row) * nSB * kQ4KB;
  __shared__ __align__(16) unsigned char hdrS[kWarpBlock][16];
  uint4* hdr = reinterpret_cast<uint4*>(hdrS[warp]);
  int qB = (lane>>4)*32 + (lane&7)*4;
  int xE = lane*4;
  unsigned nS = ((lane&15)>=8)?4u:0u;
  int hS = 8*(lane>>3);
  float sum = 0.0f;
  for (int sb=0; sb<nSB; ++sb) {
    const unsigned char* blk = rp + sb*kQ4KB;
    if (lane==0) *hdr = *reinterpret_cast<const uint4*>(blk);
    __syncwarp();
    uint4 h = *hdr;
    __syncwarp();
    float d = __half2float(__ushort_as_half((unsigned short)(h.x & 0xFFFFu)));
    float dm = __half2float(__ushort_as_half((unsigned short)(h.x >> 16)));
    unsigned a=(h.y>>hS)&0xFF, b=(h.z>>hS)&0xFF, c=(h.w>>hS)&0xFF;
    const unsigned char* qs = blk + 16;
    const float* xb = x + sb*kQKK;
    for (int g=0; g<2; ++g) {
      unsigned sc = (g==0)?(a&63u):((c&0xFu)|((a>>6)<<4));
      unsigned mn = (g==0)?(b&63u):((c>>4)|((b>>6)<<4));
      float dsc=d*float(sc), dmn=dm*float(mn);
      unsigned q4 = *reinterpret_cast<const unsigned*>(qs+g*64+qB);
      float4 xv = *reinterpret_cast<const float4*>(xb+g*128+xE);
      sum += (dsc*float((q4>>nS)&0xFu)-dmn)*xv.x;
      sum += (dsc*float((q4>>(nS+8))&0xFu)-dmn)*xv.y;
      sum += (dsc*float((q4>>(nS+16))&0xFu)-dmn)*xv.z;
      sum += (dsc*float((q4>>(nS+24))&0xFu)-dmn)*xv.w;
    }
  }
  for (int o=16; o>0; o>>=1) sum += __shfl_down_sync(0xffffffffu, sum, o);
  if (lane==0) out[row] = sum;
}

__global__ void qgemvQ6K(float* out, const unsigned char* W, const float* x, int rows, int cols) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int row = blockIdx.x * kWarpBlock + warp;
  if (row >= rows) return;
  int nSB = cols / kQKK;
  const unsigned char* rp = W + static_cast<std::size_t>(row) * nSB * kQ6KB;
  float sum = 0.0f;
  for (int sb=0; sb<nSB; ++sb) {
    const unsigned char* blk = rp + sb*kQ6KB;
    const signed char* sc = reinterpret_cast<const signed char*>(blk+192);
    float d = 0.0f;
    if (lane==0) { unsigned short h=blk[208]|(unsigned short(blk[209])<<8); d=__half2float(__ushort_as_half(h)); }
    d = __shfl_sync(0xffffffffu, d, 0);
    for (int j=0; j<16; ++j) {
      float dsc = d * float(sc[j]);
      int bo = j*8;
      int q4Lo = (blk[lane]>>(bo<32?bo:bo-32)) & 0xF;
      int hi = bo+4;
      int q4Hi = (hi<128) ? ((blk[hi/8]>>(hi%8))&0xF) : (((blk[(hi-128)/8+128]>>((hi-128)%8))>>(lane<16?0:4))&0xF);
      int q6 = q4Lo | ((q4Hi&0x3)<<4);
      sum += dsc * float(q6-32) * x[sb*kQKK+j*32+lane];
    }
  }
  for (int o=16; o>0; o>>=1) sum += __shfl_down_sync(0xffffffffu, sum, o);
  if (lane==0) out[row] = sum;
}

__global__ void f32gemv(float* out, const float* W, const float* x, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;
  float p = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) p += W[row*cols+c] * x[c];
  __shared__ float sh[256];
  sh[threadIdx.x] = p;
  __syncthreads();
  for (int s=128; s>0; s>>=1) { if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
  if (threadIdx.x==0) out[row] = sh[0];
}

void embMatmul(float* out, const EmbW& w, const float* x) {
  int g = (w.rows + kWarpBlock - 1) / kWarpBlock;
  int t = kWarpBlock * 32;
  switch (w.type) {
    case 12: qgemvQ4K<<<g,t>>>(out,(const unsigned char*)w.d,x,w.rows,w.cols); break;
    case 14: qgemvQ6K<<<g,t>>>(out,(const unsigned char*)w.d,x,w.rows,w.cols); break;
    case 8:  qgemvQ8<<<g,t>>>(out,(const unsigned char*)w.d,x,w.rows,w.cols); break;
    default: f32gemv<<<w.rows,256>>>(out,(const float*)w.d,x,w.rows,w.cols); break;
  }
}

// ---- encoder kernels -------------------------------------------------------------------------

__global__ void embedRowK(float* dst, const float* table, int token, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < d) dst[i] = table[static_cast<std::size_t>(token) * d + i];
}

__global__ void layernormK(float* out, const float* x, const float* w, const float* b, int d, float eps) {
  __shared__ float shM[256], shV[256];
  const float* row = x + blockIdx.x * d;
  float ls = 0.0f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) ls += row[i];
  shM[threadIdx.x] = ls;
  __syncthreads();
  for (int s=128; s>0; s>>=1) { if(threadIdx.x<s) shM[threadIdx.x]+=shM[threadIdx.x+s]; __syncthreads(); }
  float mean = shM[0] / d;
  float lv = 0.0f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) { float df = row[i]-mean; lv += df*df; }
  shV[threadIdx.x] = lv;
  __syncthreads();
  for (int s=128; s>0; s>>=1) { if(threadIdx.x<s) shV[threadIdx.x]+=shV[threadIdx.x+s]; __syncthreads(); }
  float inv = rsqrtf(shV[0] / d + eps);
  float* dst = out + blockIdx.x * d;
  for (int i = threadIdx.x; i < d; i += blockDim.x)
    dst[i] = (row[i] - mean) * inv * w[i] + (b ? b[i] : 0.0f);
}

__global__ void geluK(float* data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = data[i];
  data[i] = x * 0.5f * (1.0f + erff(x * 0.70710678f));
}

__global__ void addK(float* out, const float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] += x[i];
}

__global__ void mulK(float* a, const float* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] *= b[i];
}

// Bidirectional attention: each block = one head, iterates over query positions.
__global__ void attnBidirK(float* out, const float* q, const float* k, const float* v,
                           int d, int nHeads, int nSeq) {
  extern __shared__ float shS[];
  __shared__ float mx[256];
  __shared__ float sm[256];
  int h = blockIdx.x;
  if (h >= nHeads) return;
  int hd = d / nHeads, off = h * hd;
  float scale = 1.0f / sqrtf(float(hd));
  for (int i = 0; i < nSeq; ++i) {
    for (int t = threadIdx.x; t < nSeq; t += blockDim.x) {
      float dot = 0.0f;
      for (int e = 0; e < hd; ++e)
        dot += q[i*d+off+e] * k[t*d+off+e];
      shS[t] = dot * scale;
    }
    __syncthreads();
    // softmax max
    mx[threadIdx.x] = -1e30f;
    for (int t = threadIdx.x; t < nSeq; t += blockDim.x) mx[threadIdx.x] = fmaxf(mx[threadIdx.x], shS[t]);
    __syncthreads();
    for (int s=128; s>0; s>>=1) { if(threadIdx.x<s) mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+s]); __syncthreads(); }
    float m = mx[0];
    // softmax exp+sum
    float ls = 0.0f;
    for (int t = threadIdx.x; t < nSeq; t += blockDim.x) { shS[t] = expf(shS[t]-m); ls += shS[t]; }
    sm[threadIdx.x] = ls;
    __syncthreads();
    for (int s=128; s>0; s>>=1) { if(threadIdx.x<s) sm[threadIdx.x]+=sm[threadIdx.x+s]; __syncthreads(); }
    float inv = 1.0f / sm[0];
    for (int t = threadIdx.x; t < nSeq; t += blockDim.x) shS[t] *= inv;
    __syncthreads();
    // weighted sum
    for (int e = threadIdx.x; e < hd; e += blockDim.x) {
      float acc = 0.0f;
      for (int t = 0; t < nSeq; ++t) acc += shS[t] * v[t*d+off+e];
      out[i*d+off+e] = acc;
    }
    __syncthreads();
  }
}

__global__ void clsPoolK(float* out, const float* s, int d) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i < d) out[i] = s[i];
}
__global__ void meanPoolK(float* out, const float* s, int n, int d) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i >= d) return;
  float sum = 0.0f;
  for (int t = 0; t < n; ++t) sum += s[t*d+i];
  out[i] = sum / float(n);
}
__global__ void lastPoolK(float* out, const float* s, int last, int d) {
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i < d) out[i] = s[last*d+i];
}
__global__ void l2NormK(float* data, int d) {
  __shared__ float sh[256];
  float ls = 0.0f;
  for (int i = threadIdx.x; i < d; i += blockDim.x) ls += data[i]*data[i];
  sh[threadIdx.x] = ls;
  __syncthreads();
  for (int s=128; s>0; s>>=1) { if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
  float norm = sqrtf(sh[0] + 1e-12f);
  for (int i = threadIdx.x; i < d; i += blockDim.x) data[i] /= norm;
}

// ---- CudaEmbeddingModel ---------------------------------------------------------------------

struct EmbedLayer {
  float *attnNorm=nullptr, *attnNormB=nullptr;
  EmbW wq, wk, wv, wo;
  float *bq=nullptr, *bk=nullptr, *bv=nullptr, *bo=nullptr;
  float *ffnNorm=nullptr, *ffnNormB=nullptr;
  EmbW ffnUp, ffnDown;
  float *ffnUpB=nullptr, *ffnDownB=nullptr;
  EmbW ffnGate;
};

class CudaEmbedImpl : public EmbeddingModel {
 public:
  explicit CudaEmbedImpl(const EmbeddingConfig& cfg) : cfg_(cfg) {}
  ~CudaEmbedImpl() override {
    auto f = [](void* p){ if(p) cudaFree(p); };
    f(tab_.tokenEmbd); f(tab_.positionEmbd); f(tab_.tokenTypes);
    f(tab_.embdNorm); f(tab_.embdNormB);
    for (auto& L : layers_) {
      f(L.attnNorm); f(L.attnNormB); f(L.ffnNorm); f(L.ffnNormB);
      f(L.bq); f(L.bk); f(L.bv); f(L.bo); f(L.ffnUpB); f(L.ffnDownB);
      for (EmbW* w : {&L.wq,&L.wk,&L.wv,&L.wo,&L.ffnUp,&L.ffnDown,&L.ffnGate}) f(w->d);
    }
    for (void* p : scratch_) f(p);
  }

  bool init(const GpuEmbeddingTables& t, const std::vector<GpuEmbeddingLayer>& inL, std::string& err) {
    int d = cfg_.dModel, ffn = cfg_.ffn, n = cfg_.maxSeq;
    auto upF = [&](const float* h, std::size_t c) -> float* {
      float* p = nullptr;
      if (cudaMalloc(&p, c*sizeof(float)) != cudaSuccess) return nullptr;
      cudaMemcpy(p, h, c*sizeof(float), cudaMemcpyHostToDevice);
      return p;
    };
    auto upW = [&](const EmbedWeight& w, EmbW& out) -> bool {
      std::size_t bytes = embWBytes(w.ggmlType, w.rows, w.cols);
      if (!bytes) { err = "unsupported weight type"; return false; }
      if (cudaMalloc(&out.d, bytes) != cudaSuccess) { err = "cudaMalloc failed"; return false; }
      cudaMemcpy(out.d, w.host, bytes, cudaMemcpyHostToDevice);
      out.type = w.ggmlType; out.rows = w.rows; out.cols = w.cols;
      return true;
    };

    tab_.tokenEmbd = upF(t.tokenEmbd, std::size_t(cfg_.vocab)*d);
    if (!tab_.tokenEmbd) { err = "token_embd malloc failed"; return false; }
    if (t.positionEmbd) { tab_.positionEmbd = upF(t.positionEmbd, std::size_t(n)*d); if(!tab_.positionEmbd){err="pos malloc fail";return false;} }
    if (t.tokenTypes) {
      int ttCount = cfg_.tokenTypeCount > 0 ? cfg_.tokenTypeCount : 2;
      tab_.tokenTypes = upF(t.tokenTypes, std::size_t(ttCount)*d);
      if (!tab_.tokenTypes) { err = "types malloc fail"; return false; }
    }
    if (t.embdNorm) { tab_.embdNorm = upF(t.embdNorm, d); tab_.embdNormB = t.embdNormB ? upF(t.embdNormB, d) : nullptr; if(!tab_.embdNorm){err="norm malloc fail";return false;} }

    layers_.resize(inL.size());
    for (std::size_t l = 0; l < inL.size(); ++l) {
      const GpuEmbeddingLayer& s = inL[l];
      EmbedLayer& t2 = layers_[l];
      t2.attnNorm = upF(s.attnNorm, d);
      t2.attnNormB = s.attnNormB ? upF(s.attnNormB, d) : nullptr;
      t2.ffnNorm = upF(s.ffnNorm, d);
      t2.ffnNormB = s.ffnNormB ? upF(s.ffnNormB, d) : nullptr;
      if (!t2.attnNorm || !t2.ffnNorm) { err = "layer norm malloc fail"; return false; }
      if (!upW(s.wq,t2.wq)||!upW(s.wk,t2.wk)||!upW(s.wv,t2.wv)||!upW(s.wo,t2.wo)) return false;
      if (!upW(s.ffnUp,t2.ffnUp)||!upW(s.ffnDown,t2.ffnDown)) return false;
      if (s.ffnGate.valid() && !upW(s.ffnGate,t2.ffnGate)) return false;
      if (s.bq) t2.bq = upF(s.bq, d);
      if (s.bk) t2.bk = upF(s.bk, d);
      if (s.bv) t2.bv = upF(s.bv, d);
      if (s.bo) t2.bo = upF(s.bo, d);
      if (s.ffnUpB) t2.ffnUpB = upF(s.ffnUpB, ffn);
      if (s.ffnDownB) t2.ffnDownB = upF(s.ffnDownB, d);
    }

    auto sc = [&](int c) -> float* { float* p=nullptr; cudaMalloc(&p,c*sizeof(float)); cudaMemset(p,0,c*sizeof(float)); return p; };
    scratch_[0] = sc(n*d);  // states
    scratch_[1] = sc(n*d);  // norm
    scratch_[2] = sc(n*d);  // q
    scratch_[3] = sc(n*d);  // k
    scratch_[4] = sc(n*d);  // v
    scratch_[5] = sc(n*d);  // attn
    scratch_[6] = sc(n*d);  // tmp
    scratch_[7] = sc(n*ffn); // ffn
    scratch_[8] = cfg_.ffnGated ? sc(n*ffn) : nullptr; // ffnGate
    for (int i = 0; i < 9; ++i)
      if (scratch_[i] == nullptr) { err = "scratch malloc fail"; return false; }
    return true;
  }

  bool embed(const std::vector<int>& tokens, std::vector<float>& out, std::string& err) override {
    return embedWith(tokens, static_cast<int>(cfg_.defaultPooling), cfg_.defaultNormalize, out, err);
  }

  bool embedWith(const std::vector<int>& tokens, int pooling, bool normalize,
                 std::vector<float>& out, std::string& err) override {
    err.clear();
    int n = static_cast<int>(tokens.size()), d = cfg_.dModel;
    if (n == 0) { err = "empty sequence"; return false; }
    if (n > cfg_.maxSeq) { err = "too long"; return false; }

    int g = [](int x){return (x+255)/256;};
    int g256 = g(d);

    // Embeddings
    for (int t = 0; t < n; ++t)
      embedRowK<<<g256, kEmbThreads>>>(scratch_[0]+t*d, tab_.tokenEmbd, tokens[t], d);
    if (tab_.positionEmbd)
      for (int t = 0; t < n; ++t) {
        embedRowK<<<g256, kEmbThreads>>>(scratch_[6], tab_.positionEmbd, t, d);
        addK<<<g256, kEmbThreads>>>(scratch_[0]+t*d, scratch_[6], d);
      }
    if (tab_.tokenTypes)
      for (int t = 0; t < n; ++t) {
        embedRowK<<<g256, kEmbThreads>>>(scratch_[6], tab_.tokenTypes, 0, d);
        addK<<<g256, kEmbThreads>>>(scratch_[0]+t*d, scratch_[6], d);
      }
    if (tab_.embdNorm)
      layernormK<<<n, kEmbThreads>>>(scratch_[0], scratch_[0], tab_.embdNorm, tab_.embdNormB, d, cfg_.normEps);

    // Transformer layers
    for (auto& L : layers_) {
      // Post-norm copy
      cudaMemcpy(scratch_[1], scratch_[0], std::size_t(n)*d*sizeof(float), cudaMemcpyDeviceToDevice);

      // Q, K, V
      for (int r = 0; r < n; ++r) {
        embMatmul(scratch_[2]+r*d, L.wq, scratch_[1]+r*d);
        embMatmul(scratch_[3]+r*d, L.wk, scratch_[1]+r*d);
        embMatmul(scratch_[4]+r*d, L.wv, scratch_[1]+r*d);
        if (L.bq) addK<<<g(d),kEmbThreads>>>(scratch_[2]+r*d, L.bq, d);
        if (L.bk) addK<<<g(d),kEmbThreads>>>(scratch_[3]+r*d, L.bk, d);
        if (L.bv) addK<<<g(d),kEmbThreads>>>(scratch_[4]+r*d, L.bv, d);
      }

      // Bidirectional attention
      attnBidirK<<<cfg_.nHeads, kEmbThreads, n*sizeof(float)>>>(
          scratch_[5], scratch_[2], scratch_[3], scratch_[4], d, cfg_.nHeads, n);

      // O-proj + residual + post-norm
      for (int r = 0; r < n; ++r) {
        embMatmul(scratch_[6]+r*d, L.wo, scratch_[5]+r*d);
        if (L.bo) addK<<<g(d),kEmbThreads>>>(scratch_[6]+r*d, L.bo, d);
      }
      for (int r = 0; r < n; ++r) {
        addK<<<g(d),kEmbThreads>>>(scratch_[0]+r*d, scratch_[6]+r*d, d);
        layernormK<<<1,kEmbThreads>>>(scratch_[0]+r*d, scratch_[0]+r*d, L.attnNorm, L.attnNormB, d, cfg_.normEps);
      }

      // FFN
      int ffn = cfg_.ffn;
      for (int r = 0; r < n; ++r)
        embMatmul(scratch_[7]+r*ffn, L.ffnUp, scratch_[0]+r*d);
      for (int r = 0; r < n; ++r)
        if (L.ffnUpB) addK<<<g(ffn),kEmbThreads>>>(scratch_[7]+r*ffn, L.ffnUpB, ffn);

      if (cfg_.ffnGated) {
        for (int r = 0; r < n; ++r)
          embMatmul(scratch_[8]+r*ffn, L.ffnGate, scratch_[0]+r*d);
        geluK<<<g(n*ffn),kEmbThreads>>>(scratch_[8], n*ffn);
        mulK<<<g(n*ffn),kEmbThreads>>>(scratch_[7], scratch_[8], n*ffn);
      } else {
        geluK<<<g(n*ffn),kEmbThreads>>>(scratch_[7], n*ffn);
      }

      for (int r = 0; r < n; ++r)
        embMatmul(scratch_[6]+r*d, L.ffnDown, scratch_[7]+r*ffn);
      for (int r = 0; r < n; ++r)
        if (L.ffnDownB) addK<<<g(d),kEmbThreads>>>(scratch_[6]+r*d, L.ffnDownB, d);
      for (int r = 0; r < n; ++r) {
        addK<<<g(d),kEmbThreads>>>(scratch_[0]+r*d, scratch_[6]+r*d, d);
        layernormK<<<1,kEmbThreads>>>(scratch_[0]+r*d, scratch_[0]+r*d, L.ffnNorm, L.ffnNormB, d, cfg_.normEps);
      }
    }

    // Pooling: run into device scratch buffer, then copy to host
    switch (pooling) {
      case 0: clsPoolK<<<g(d),kEmbThreads>>>(scratch_[6], scratch_[0], d); break;
      case 3: lastPoolK<<<g(d),kEmbThreads>>>(scratch_[6], scratch_[0], n-1, d); break;
      case 1: case 2: default: meanPoolK<<<g(d),kEmbThreads>>>(scratch_[6], scratch_[0], n, d); break;
    }
    if (normalize) l2NormK<<<1,kEmbThreads>>>(scratch_[6], d);
    out.resize(d);
    cudaMemcpy(out.data(), scratch_[6], d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return true;
  }

  bool embedBatch(const std::vector<std::vector<int>>& batch,
                  std::vector<std::vector<float>>& out, std::string& err) override {
    out.resize(batch.size());
    for (std::size_t i = 0; i < batch.size(); ++i) {
      if (!embed(batch[i], out[i], err)) return false;
    }
    return true;
  }

  bool embedTokens(const std::vector<int>& tokens, std::vector<float>& out, std::string& err) override {
    err.clear();
    int n = static_cast<int>(tokens.size()), d = cfg_.dModel;
    if (n == 0) { err = "empty sequence"; return false; }
    if (n > cfg_.maxSeq) { err = "too long"; return false; }
    int g = [](int x){return (x+255)/256;};
    // Embeddings (same as embed)
    for (int t = 0; t < n; ++t)
      embedRowK<<<g(d),kEmbThreads>>>(scratch_[0]+t*d, tab_.tokenEmbd, tokens[t], d);
    if (tab_.positionEmbd)
      for (int t = 0; t < n; ++t) {
        embedRowK<<<g(d),kEmbThreads>>>(scratch_[6], tab_.positionEmbd, t, d);
        addK<<<g(d),kEmbThreads>>>(scratch_[0]+t*d, scratch_[6], d);
      }
    if (tab_.embdNorm)
      layernormK<<<n,kEmbThreads>>>(scratch_[0], scratch_[0], tab_.embdNorm, tab_.embdNormB, d, cfg_.normEps);
    // Layers (same as embed, skip pooling)
    for (auto& L : layers_) {
      cudaMemcpy(scratch_[1], scratch_[0], std::size_t(n)*d*sizeof(float), cudaMemcpyDeviceToDevice);
      for (int r = 0; r < n; ++r) {
        embMatmul(scratch_[2]+r*d, L.wq, scratch_[1]+r*d);
        embMatmul(scratch_[3]+r*d, L.wk, scratch_[1]+r*d);
        embMatmul(scratch_[4]+r*d, L.wv, scratch_[1]+r*d);
        if (L.bq) addK<<<g(d),kEmbThreads>>>(scratch_[2]+r*d, L.bq, d);
        if (L.bk) addK<<<g(d),kEmbThreads>>>(scratch_[3]+r*d, L.bk, d);
        if (L.bv) addK<<<g(d),kEmbThreads>>>(scratch_[4]+r*d, L.bv, d);
      }
      attnBidirK<<<cfg_.nHeads,kEmbThreads,n*sizeof(float)>>>(
          scratch_[5], scratch_[2], scratch_[3], scratch_[4], d, cfg_.nHeads, n);
      for (int r = 0; r < n; ++r) {
        embMatmul(scratch_[6]+r*d, L.wo, scratch_[5]+r*d);
        if (L.bo) addK<<<g(d),kEmbThreads>>>(scratch_[6]+r*d, L.bo, d);
      }
      for (int r = 0; r < n; ++r) {
        addK<<<g(d),kEmbThreads>>>(scratch_[0]+r*d, scratch_[6]+r*d, d);
        layernormK<<<1,kEmbThreads>>>(scratch_[0]+r*d, scratch_[0]+r*d, L.attnNorm, L.attnNormB, d, cfg_.normEps);
      }
      int ffn = cfg_.ffn;
      for (int r = 0; r < n; ++r)
        embMatmul(scratch_[7]+r*ffn, L.ffnUp, scratch_[0]+r*d);
      if (cfg_.ffnGated) {
        for (int r = 0; r < n; ++r)
          embMatmul(scratch_[8]+r*ffn, L.ffnGate, scratch_[0]+r*d);
        geluK<<<g(n*ffn),kEmbThreads>>>(scratch_[8], n*ffn);
        mulK<<<g(n*ffn),kEmbThreads>>>(scratch_[7], scratch_[8], n*ffn);
      } else {
        geluK<<<g(n*ffn),kEmbThreads>>>(scratch_[7], n*ffn);
      }
      for (int r = 0; r < n; ++r)
        embMatmul(scratch_[6]+r*d, L.ffnDown, scratch_[7]+r*ffn);
      for (int r = 0; r < n; ++r) {
        addK<<<g(d),kEmbThreads>>>(scratch_[0]+r*d, scratch_[6]+r*d, d);
        layernormK<<<1,kEmbThreads>>>(scratch_[0]+r*d, scratch_[0]+r*d, L.ffnNorm, L.ffnNormB, d, cfg_.normEps);
      }
    }
    // Copy raw hidden states to host
    out.resize(std::size_t(n) * d);
    cudaMemcpy(out.data(), scratch_[0], std::size_t(n)*d*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return true;
  }

  int dim() const override { return cfg_.dModel; }
  int maxSeqLen() const override { return cfg_.maxSeq; }
  int defaultPooling() const override { return static_cast<int>(cfg_.defaultPooling); }
  bool defaultNormalize() const override { return cfg_.defaultNormalize; }
  std::string backendName() const override { return "cuda"; }

 private:
  EmbeddingConfig cfg_;
  GpuEmbeddingTables tab_{};
  std::vector<EmbedLayer> layers_;
  static constexpr int kScratchCount = 9;
  float* scratch_[kScratchCount] = {};
};

}  // namespace

std::unique_ptr<EmbeddingModel> createEmbeddingModel(
    const EmbeddingConfig& cfg, const GpuEmbeddingTables& tables,
    const std::vector<GpuEmbeddingLayer>& layers, std::string& error) {
  auto m = std::make_unique<CudaEmbedImpl>(cfg);
  if (!m->init(tables, layers, error)) return nullptr;
  return m;
}

}  // namespace qorvix::cuda
