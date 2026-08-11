#pragma once

// Sequence-level reductions and the vector-similarity metric shared by the embeddings engine and
// the RAG vector store (SPEC "EMBEDDINGS ENGINE" / "RAG SYSTEM").
//
// Deliberately kept out of ops.hpp, which is scoped to the transformer primitives themselves — a
// pooling step collapses a whole sequence and a cosine is a metric, neither of which is part of a
// forward pass. Same `ops` namespace, so callers see one namespace either way.
//
// `states` is always [nTokens, dim] row-major: row t is the hidden state of token t.
namespace qorvix::runtime::ops {

// out[dim] = mean over all token rows. The pooling every sentence-transformers model that isn't
// CLS-pooled uses (all-MiniLM-L6-v2, most of the MTEB set).
void meanPool(float* out, const float* states, int nTokens, int dim);

// out[dim] = row 0, the [CLS] token's state (bge-* family).
void clsPool(float* out, const float* states, int nTokens, int dim);

// out[dim] = row nTokens-1, the last token's state (decoder-style embedding models).
void lastPool(float* out, const float* states, int nTokens, int dim);

// Scales x to unit L2 norm in place. A zero (or denormal-norm) vector is LEFT UNTOUCHED rather
// than divided by zero: returning NaN here is how one empty document silently poisons an entire
// index, since every later cosine against it is NaN and every comparison with NaN is false.
void l2Normalize(float* x, int n);

// L2 norm of x, without modifying it.
float l2Norm(const float* x, int n);

// Cosine similarity in [-1, 1]; 0 when either vector has zero norm (see l2Normalize).
//
// Uses cpu::dotProductF32 for all three sums, so it inherits the runtime AVX2/NEON dispatch. This
// is the inner loop of every RAG search — a brute-force top-k over 100k chunks calls it 100k times
// per query — so it must not be a naive scalar loop.
float cosineSimilarity(const float* a, const float* b, int n);

}  // namespace qorvix::runtime::ops
