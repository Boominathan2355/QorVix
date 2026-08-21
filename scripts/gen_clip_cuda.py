#!/usr/bin/env python3
"""Generate the CUDA CLIP vision model implementation."""
import os

OUT = os.path.join(os.path.dirname(__file__), '..', 'cuda', 'src', 'clip_model.cu')

code = r'''// CUDA CLIP vision tower (Phase 11c) - GPU implementation of the vision encoder.
// Pre-norm ViT with quick-GELU, patch embedding via matmul, learned positions.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "qorvix/cuda/clip_model.hpp"

namespace qorvix::cuda {
namespace {

constexpr int kClipThreads = 256;
constexpr int kWarpBlock = 8;
constexpr int kQKK = 256;
constexpr int kQ4KB = 144;
constexpr int kQ6KB = 210;

std::size_t clipWBytes(std::uint32_t t, int r, int c) {
  std::size_t n = static_cast<std::size_t>(r) * c;
  switch (t) {
    case 0: return n * 4;
    case 8: return n / 32 * 34;
    case 12: return n / kQKK * kQ4KB;
    case 14: return n / kQKK * kQ6KB;
    default: return 0;
  }
}

struct ClipDevW { void* d = nullptr; std::uint32_t type = 0; int rows = 0; int cols = 0; };

// ---- quantized GEMV (same as embedding_model.cu) ----
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

void clipMatmul(float* out, const ClipDevW& w, const float* x) {
  int g = (w.rows + kWarpBlock - 1) / kWarpBlock;
  int t = kWarpBlock * 32;
  switch (w.type) {
    case 12: qgemvQ4K<<<g,t>>>(out,(const unsigned char*)w.d,x,w.rows,w.cols); break;
    case 14: qgemvQ6K<<<g,t>>>(out,(const unsigned char*)w.d,x,w.rows,w.cols); break;
    case 8:  qgemvQ8<<<g,t>>>(out,(const unsigned char*)w.d,x,w.rows,w.cols); break;
    default: f32gemv<<<w.rows,256>>>(out,(const float*)w.d,x,w.rows,w.cols); break;
  }
}

// ---- CLIP-specific kernels ----

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

__global__ void geluQuickK(float* data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = data[i];
  data[i] = x / (1.0f + expf(-1.702f * x));
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

// Bidirectional attention (same as embedding engine)
__global__ void attnBidirK(float* out, const float* q, const float* k, const float* v,
                           int d, int nHeads, int nSeq) {
  extern __shared__ float shS[];
  int h = blockIdx.x;
  if (h >= nHeads) return;
  int hd = d / nHeads, off = h * hd;
  float scale = 1.0f / sqrtf(float(hd));
  for (int i = 0; i < nSeq; ++i) {
    for (int t = threadIdx.x; t < nSeq; t += blockDim.x) {
      float dot = 0.0f;
      for (int e = 0; e < hd; ++e) dot += q[i*d+off+e] * k[t*d+off+e];
      shS[t] = dot * scale;
    }
    __syncthreads();
    __shared__ float mx[256];
    mx[threadIdx.x] = -1e30f;
    for (int t = threadIdx.x; t < nSeq; t += blockDim.x) mx[threadIdx.x] = fmaxf(mx[threadIdx.x], shS[t]);
    __syncthreads();
    for (int s=128; s>0; s>>=1) { if(threadIdx.x<s) mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+s]); __syncthreads(); }
    float m = mx[0];
    float ls = 0.0f;
    for (int t = threadIdx.x; t < nSeq; t += blockDim.x) { shS[t] = expf(shS[t]-m); ls += shS[t]; }
    __shared__ float sm[256];
    sm[threadIdx.x] = ls;
    __syncthreads();
    for (int s=128; s>0; s>>=1) { if(threadIdx.x<s) sm[threadIdx.x]+=sm[threadIdx.x+s]; __syncthreads(); }
    float inv = 1.0f / sm[0];
    for (int t = threadIdx.x; t < nSeq; t += blockDim.x) shS[t] *= inv;
    __syncthreads();
    for (int e = threadIdx.x; e < hd; e += blockDim.x) {
      float acc = 0.0f;
      for (int t = 0; t < nSeq; ++t) acc += shS[t] * v[t*d+off+e];
      out[i*d+off+e] = acc;
    }
    __syncthreads();
  }
}

// ---- CudaClipVisionModel ----

struct ClipDevLayer {
  ClipDevW wq, wk, wv, wo;
  float *bq=nullptr, *bk=nullptr, *bv=nullptr, *bo=nullptr;
  float *ln1W=nullptr, *ln1B=nullptr, *ln2W=nullptr, *ln2B=nullptr;
  ClipDevW ffnExpand, ffnContract;
  float *ffnExpandB=nullptr, *ffnContractB=nullptr;
};

class CudaClipImpl : public ClipVisionModel {
 public:
  explicit CudaClipImpl(const ClipConfig& cfg) : cfg_(cfg) {}
  ~CudaClipImpl() override {
    auto f = [](void* p){ if(p) cudaFree(p); };
    f(wtab_.classEmbd);
    for (auto& L : layers_) {
      f(L.bq); f(L.bk); f(L.bv); f(L.bo);
      f(L.ln1W); f(L.ln1B); f(L.ln2W); f(L.ln2B);
      f(L.ffnExpandB); f(L.ffnContractB);
      for (ClipDevW* w : {&L.wq,&L.wk,&L.wv,&L.wo,&L.ffnExpand,&L.ffnContract}) f(w->d);
    }
    for (void* p : {wtab_.patchEmbd.d, wtab_.positionEmbd.d, wtab_.preLnW, wtab_.preLnB, wtab_.postLnW, wtab_.postLnB, wtab_.mm0.d, wtab_.mm0B, wtab_.mm2.d, wtab_.mm2B}) f(p);
    for (void* p : dScratch_) f(p);
  }

  bool init(const GpuClipWeights& w, const std::vector<GpuClipLayer>& inL, std::string& err) {
    int d = cfg_.embeddingLength, ffn = cfg_.feedForwardLength, n = cfg_.tokenCount();
    auto upF = [&](const float* h, std::size_t c) -> float* {
      float* p = nullptr;
      if (cudaMalloc(&p, c*sizeof(float)) != cudaSuccess) return nullptr;
      cudaMemcpy(p, h, c*sizeof(float), cudaMemcpyHostToDevice);
      return p;
    };
    auto upW = [&](const GpuClipWeight& ww, ClipDevW& out) -> bool {
      std::size_t bytes = clipWBytes(ww.ggmlType, ww.rows, ww.cols);
      if (!bytes) { err = "unsupported weight type"; return false; }
      if (cudaMalloc(&out.d, bytes) != cudaSuccess) { err = "cudaMalloc failed"; return false; }
      cudaMemcpy(out.d, ww.host, bytes, cudaMemcpyHostToDevice);
      out.type = ww.ggmlType; out.rows = ww.rows; out.cols = ww.cols;
      return true;
    };

    // Upload patch embedding weight
    if (!upW(w.patchEmbd, wtab_.patchEmbd)) { err = "patch_embd failed"; return false; }
    if (w.classEmbd) wtab_.classEmbd = upF(w.classEmbd, d);
    if (w.positionEmbd.valid()) {
      if (!upW(w.positionEmbd, wtab_.positionEmbd)) { err = "pos_embd failed"; return false; }
    }
    if (w.preLnW) { wtab_.preLnW = upF(w.preLnW, d); wtab_.preLnB = w.preLnB ? upF(w.preLnB, d) : nullptr; }
    if (w.postLnW) { wtab_.postLnW = upF(w.postLnW, d); wtab_.postLnB = w.postLnB ? upF(w.postLnB, d) : nullptr; }
    if (w.mm0.valid()) {
      if (!upW(w.mm0, wtab_.mm0)) return false;
      wtab_.mm0B = w.mm0B ? upF(w.mm0B, w.mm0.rows) : nullptr;
    }
    if (w.mm2.valid()) {
      if (!upW(w.mm2, wtab_.mm2)) return false;
      wtab_.mm2B = w.mm2B ? upF(w.mm2B, w.mm2.rows) : nullptr;
    }

    layers_.resize(inL.size());
    for (std::size_t l = 0; l < inL.size(); ++l) {
      const GpuClipLayer& s = inL[l];
      ClipDevLayer& t = layers_[l];
      if (!upW(s.wq,t.wq)||!upW(s.wk,t.wk)||!upW(s.wv,t.wv)||!upW(s.wo,t.wo)) return false;
      if (!upW(s.ffnExpand,t.ffnExpand)||!upW(s.ffnContract,t.ffnContract)) return false;
      t.bq = s.bq ? upF(s.bq, d) : nullptr;
      t.bk = s.bk ? upF(s.bk, d) : nullptr;
      t.bv = s.bv ? upF(s.bv, d) : nullptr;
      t.bo = s.bo ? upF(s.bo, d) : nullptr;
      t.ln1W = upF(s.ln1W, d); t.ln1B = s.ln1B ? upF(s.ln1B, d) : nullptr;
      t.ln2W = upF(s.ln2W, d); t.ln2B = s.ln2B ? upF(s.ln2B, d) : nullptr;
      t.ffnExpandB = s.ffnExpandB ? upF(s.ffnExpandB, ffn) : nullptr;
      t.ffnContractB = s.ffnContractB ? upF(s.ffnContractB, d) : nullptr;
    }

    auto sc = [&](int c) -> float* { float* p=nullptr; cudaMalloc(&p,c*sizeof(float)); cudaMemset(p,0,c*sizeof(float)); return p; };
    dScratch_[0] = sc(n*d);  // states
    dScratch_[1] = sc(n*d);  // norm
    dScratch_[2] = sc(n*d);  // q
    dScratch_[3] = sc(n*d);  // k
    dScratch_[4] = sc(n*d);  // v
    dScratch_[5] = sc(n*d);  // attn
    dScratch_[6] = sc(std::max(d, ffn)*n);  // tmp
    dScratch_[7] = sc(n*ffn);  // ffn
    for (int i = 0; i < 8; ++i)
      if (!dScratch_[i]) { err = "scratch malloc fail"; return false; }
    return true;
  }

  bool encodePixels(const std::vector<float>& chw, std::vector<float>& out, std::string& err) override {
    err.clear();
    int d = cfg_.embeddingLength, ffn = cfg_.feedForwardLength;
    int side = cfg_.patchesPerSide(), patch = cfg_.patchSize, imgSize = cfg_.imageSize;
    int n = cfg_.tokenCount();
    float eps = cfg_.normEpsilon;
    int g = [](int x){return (x+255)/256;};

    // Patch embedding: flatten + matmul
    int patchElems = patch * patch * 3;
    std::vector<float> flat(static_cast<std::size_t>(side)*side*patchElems);
    std::size_t plane = static_cast<std::size_t>(imgSize)*imgSize;
    for (int py = 0; py < side; ++py)
      for (int px = 0; px < side; ++px) {
        float* dst = flat.data() + (static_cast<std::size_t>(py)*side+px)*patchElems;
        for (int c = 0; c < 3; ++c)
          for (int ky = 0; ky < patch; ++ky)
            for (int kx = 0; kx < patch; ++kx)
              dst[static_cast<std::size_t>(c)*patch*patch+static_cast<std::size_t>(ky)*patch+kx] =
                  chw[c*plane + static_cast<std::size_t>(py*patch+ky)*imgSize + px*patch+kx];
      }

    // Upload flat patches and run patchEmbd matmul
    float* dFlat = nullptr;
    cudaMalloc(&dFlat, flat.size()*sizeof(float));
    cudaMemcpy(dFlat, flat.data(), flat.size()*sizeof(float), cudaMemcpyHostToDevice);
    clipMatmul(dScratch_[0]+d, wtab_.patchEmbd, dFlat);
    cudaFree(dFlat);

    // Copy class embedding to row 0
    if (wtab_.classEmbd)
      cudaMemcpy(dScratch_[0], wtab_.classEmbd, d*sizeof(float), cudaMemcpyDeviceToDevice);

    // Position embeddings
    if (wtab_.positionEmbd.d)
      for (int t = 0; t < n; ++t) {
        embedRowK<<<g(d),kClipThreads>>>(dScratch_[6], (const float*)wtab_.positionEmbd.d, t, d);
        addK<<<g(d),kClipThreads>>>(dScratch_[0]+t*d, dScratch_[6], d);
      }

    // Pre-norm
    if (wtab_.preLnW)
      for (int t = 0; t < n; ++t)
        layernormK<<<1,kClipThreads>>>(dScratch_[0]+t*d, dScratch_[0]+t*d, wtab_.preLnW, wtab_.preLnB, d, eps);

    // Pre-norm transformer blocks
    for (auto& L : layers_) {
      // Pre-norm -> attention -> residual
      for (int t = 0; t < n; ++t)
        layernormK<<<1,kClipThreads>>>(dScratch_[1]+t*d, dScratch_[0]+t*d, L.ln1W, L.ln1B, d, eps);
      // Q, K, V
      for (int r = 0; r < n; ++r) {
        clipMatmul(dScratch_[2]+r*d, L.wq, dScratch_[1]+r*d);
        clipMatmul(dScratch_[3]+r*d, L.wk, dScratch_[1]+r*d);
        clipMatmul(dScratch_[4]+r*d, L.wv, dScratch_[1]+r*d);
        if (L.bq) addK<<<g(d),kClipThreads>>>(dScratch_[2]+r*d, L.bq, d);
        if (L.bk) addK<<<g(d),kClipThreads>>>(dScratch_[3]+r*d, L.bk, d);
        if (L.bv) addK<<<g(d),kClipThreads>>>(dScratch_[4]+r*d, L.bv, d);
      }
      attnBidirK<<<cfg_.headCount,kClipThreads,n*sizeof(float)>>>(
          dScratch_[5], dScratch_[2], dScratch_[3], dScratch_[4], d, cfg_.headCount, n);
      for (int r = 0; r < n; ++r) {
        clipMatmul(dScratch_[6]+r*d, L.wo, dScratch_[5]+r*d);
        if (L.bo) addK<<<g(d),kClipThreads>>>(dScratch_[6]+r*d, L.bo, d);
      }
      for (int r = 0; r < n; ++r)
        addK<<<g(d),kClipThreads>>>(dScratch_[0]+r*d, dScratch_[6]+r*d, d);

      // Pre-norm -> FFN -> residual
      for (int t = 0; t < n; ++t)
        layernormK<<<1,kClipThreads>>>(dScratch_[1]+t*d, dScratch_[0]+t*d, L.ln2W, L.ln2B, d, eps);
      for (int r = 0; r < n; ++r)
        clipMatmul(dScratch_[7]+r*ffn, L.ffnExpand, dScratch_[1]+r*d);
      for (int r = 0; r < n; ++r)
        if (L.ffnExpandB) addK<<<g(ffn),kClipThreads>>>(dScratch_[7]+r*ffn, L.ffnExpandB, ffn);
      if (cfg_.useGelu)
        geluK<<<g(n*ffn),kClipThreads>>>(dScratch_[7], n*ffn);
      else
        geluQuickK<<<g(n*ffn),kClipThreads>>>(dScratch_[7], n*ffn);
      for (int r = 0; r < n; ++r)
        clipMatmul(dScratch_[6]+r*d, L.ffnContract, dScratch_[7]+r*ffn);
      for (int r = 0; r < n; ++r)
        if (L.ffnContractB) addK<<<g(d),kClipThreads>>>(dScratch_[6]+r*d, L.ffnContractB, d);
      for (int r = 0; r < n; ++r)
        addK<<<g(d),kClipThreads>>>(dScratch_[0]+r*d, dScratch_[6]+r*d, d);
    }

    // Post-norm
    if (wtab_.postLnW)
      for (int t = 0; t < n; ++t)
        layernormK<<<1,kClipThreads>>>(dScratch_[0]+t*d, dScratch_[0]+t*d, wtab_.postLnW, wtab_.postLnB, d, eps);

    // Drop class token: return rows 1..N (patch tokens only)
    int patches = n - 1;
    out.resize(static_cast<std::size_t>(patches) * d);
    cudaMemcpy(out.data(), dScratch_[0]+d, static_cast<std::size_t>(patches)*d*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return true;
  }

  bool project(const std::vector<float>& hidden, std::vector<float>& out, std::string& err) override {
    err.clear();
    if (!cfg_.hasProjector) { err = "no projector"; return false; }
    int d = cfg_.embeddingLength, tokens = patchTokens(), llm = cfg_.projectedDim;
    if ((int)hidden.size() != tokens*d) { err = "wrong shape"; return false; }
    int g = [](int x){return (x+255)/256;};

    float* dH = nullptr;
    cudaMalloc(&dH, hidden.size()*sizeof(float));
    cudaMemcpy(dH, hidden.data(), hidden.size()*sizeof(float), cudaMemcpyHostToDevice);

    // mm0: Linear -> GELU
    float* dMid = nullptr;
    cudaMalloc(&dMid, static_cast<std::size_t>(tokens)*llm*sizeof(float));
    for (int r = 0; r < tokens; ++r)
      clipMatmul(dMid+r*llm, wtab_.mm0, dH+r*d);
    if (wtab_.mm0B)
      for (int r = 0; r < tokens; ++r)
        addK<<<g(llm),kClipThreads>>>(dMid+r*llm, wtab_.mm0B, llm);
    geluK<<<g(tokens*llm),kClipThreads>>>(dMid, tokens*llm);

    // mm2: Linear
    float* dOut = nullptr;
    cudaMalloc(&dOut, static_cast<std::size_t>(tokens)*llm*sizeof(float));
    cudaMemset(dOut, 0, static_cast<std::size_t>(tokens)*llm*sizeof(float));
    for (int r = 0; r < tokens; ++r)
      clipMatmul(dOut+r*llm, wtab_.mm2, dMid+r*llm);
    if (wtab_.mm2B)
      for (int r = 0; r < tokens; ++r)
        addK<<<g(llm),kClipThreads>>>(dOut+r*llm, wtab_.mm2B, llm);

    out.resize(static_cast<std::size_t>(tokens)*llm);
    cudaMemcpy(out.data(), dOut, out.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dH); cudaFree(dMid); cudaFree(dOut);
    cudaDeviceSynchronize();
    return true;
  }

  int embeddingLength() const override { return cfg_.embeddingLength; }
  int patchTokens() const override { return cfg_.tokenCount() - 1; }
  int projectedDim() const override { return cfg_.projectedDim; }
  bool hasProjector() const override { return cfg_.hasProjector; }
  std::string backendName() const override { return "cuda"; }

 private:
  ClipConfig cfg_;
  GpuClipWeights wtab_{};
  std::vector<ClipDevLayer> layers_;
  float* dScratch_[8] = {};
};

}  // namespace

std::unique_ptr<ClipVisionModel> createClipVisionModel(
    const ClipConfig& cfg, const GpuClipWeights& weights,
    const std::vector<GpuClipLayer>& layers, std::string& error) {
  auto m = std::make_unique<CudaClipImpl>(cfg);
  if (!m->init(weights, layers, error)) return nullptr;
  return m;
}

}  // namespace qorvix::cuda
'''

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', newline='\n') as f:
    f.write(code)
print(f"Written {len(code)} bytes to {OUT}")
