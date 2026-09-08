// Fused normalized-sigmoid routing from precomputed expert indices. One warp handles
// each token and one lane handles each expert slot, requiring topk <= 32.
//
// Matches topk_routing_with_score_function(score_function="sigmoid", dense_output=False):
//   routing_probs[t, e] = scaling * sigmoid(logits[t, e]) / (sum_k sigmoid(...) + 1e-20)
//                         for e in indices[t], else 0
//   routing_map[t, e]   = e in indices[t]
// Expert bias affects selection only, which replay replaces.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {

constexpr int kWarp = 32;
constexpr int kThreadsPerBlock = 256;  // 8 tokens per block
constexpr float kEps = 1e-20f;         // matches Megatron's normalization epsilon
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor_sync(kFullMask, v, offset);
  }
  return v;
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

// scores and denom are saved for backward; they are [num_tokens, topk] and [num_tokens],
// i.e. topk/num_experts of the memory Megatron's autograd graph holds for the same step.
template <typename scalar_t>
__global__ void replay_fwd_dense_kernel(const scalar_t* __restrict__ logits,
                                        const int* __restrict__ indices,
                                        scalar_t* __restrict__ routing_probs,
                                        bool* __restrict__ routing_map,
                                        float* __restrict__ scores_out,
                                        float* __restrict__ denom_out, int num_tokens,
                                        int num_experts, int topk, float scaling) {
  extern __shared__ char smem_raw[];
  const int warps_per_block = blockDim.x / kWarp;
  const int warp_in_block = threadIdx.x / kWarp;
  const int lane = threadIdx.x % kWarp;

  float* prob_s = reinterpret_cast<float*>(smem_raw);
  int* idx_s = reinterpret_cast<int*>(smem_raw + warps_per_block * kWarp * sizeof(float));

  const int token = blockIdx.x * warps_per_block + warp_in_block;
  const int slot = warp_in_block * kWarp + lane;
  const long compact = static_cast<long>(token) * topk + lane;

  const bool active = token < num_tokens && lane < topk;
  int expert = -1;
  if (active) {
    expert = indices[compact];
  }
  // Fail the whole warp before any full-mask shuffle. This matches Megatron's gather
  // failure instead of silently dropping an invalid replayed route.
  const bool valid = !active || (expert >= 0 && expert < num_experts);
  CUDA_KERNEL_ASSERT(__ballot_sync(kFullMask, valid) == kFullMask);

  float score = 0.0f;
  if (active) {
    score = sigmoidf(static_cast<float>(logits[static_cast<long>(token) * num_experts + expert]));
  }
  const float total = warp_sum(score) + kEps;
  prob_s[slot] = active ? scaling * score / total : 0.0f;
  idx_s[slot] = active ? expert : -1;
  if (active) {
    scores_out[compact] = score;
  }
  if (lane == 0 && token < num_tokens) {
    denom_out[token] = total;
  }
  __syncwarp();

  if (token >= num_tokens) return;
  const int base = warp_in_block * kWarp;
  const long row = static_cast<long>(token) * num_experts;
  for (int e = lane; e < num_experts; e += kWarp) {
    float value = 0.0f;
    bool hit = false;
    for (int k = 0; k < topk; ++k) {
      if (idx_s[base + k] == e) {
        value = prob_s[base + k];
        hit = true;
      }
    }
    routing_probs[row + e] = static_cast<scalar_t>(value);
    routing_map[row + e] = hit;
  }
}

//   p_j     = s_j / S                       (unscaled normalized prob)
//   g_j     = scaling * grad_routing_probs[t, idx_j]
//   dL/ds_j = (g_j - sum_i g_i * p_i) / S
//   dL/dl_j = dL/ds_j * s_j * (1 - s_j)
// Every dense grad_logits element is written. Contributions from duplicated replay
// indices are summed in shared memory, matching Megatron's gather/scatter-add autograd
// without requiring dtype-specific atomics.
template <typename scalar_t>
__global__ void replay_bwd_dense_kernel(const scalar_t* __restrict__ grad_routing_probs,
                                        const float* __restrict__ scores,
                                        const float* __restrict__ denom,
                                        const int* __restrict__ indices,
                                        scalar_t* __restrict__ grad_logits, int num_tokens,
                                        int num_experts, int topk, float scaling) {
  extern __shared__ char smem_raw[];
  const int warps_per_block = blockDim.x / kWarp;
  const int warp_in_block = threadIdx.x / kWarp;
  const int slot = warp_in_block * kWarp + threadIdx.x % kWarp;
  float* grad_s = reinterpret_cast<float*>(smem_raw);
  int* idx_s = reinterpret_cast<int*>(smem_raw + warps_per_block * kWarp * sizeof(float));

  const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int token = global_thread / kWarp;
  const int lane = global_thread % kWarp;

  const long compact = static_cast<long>(token) * topk + lane;
  const bool active = token < num_tokens && lane < topk;
  int expert = -1;
  if (active) {
    expert = indices[compact];
  }
  const bool valid = !active || (expert >= 0 && expert < num_experts);
  CUDA_KERNEL_ASSERT(__ballot_sync(kFullMask, valid) == kFullMask);

  float grad_prob = 0.0f;
  float score = 0.0f;
  if (active) {
    grad_prob = scaling * grad_routing_probs[static_cast<long>(token) * num_experts + expert];
    score = scores[compact];
  }
  const float total = token < num_tokens ? denom[token] : 1.0f;
  const float prob = score / total;
  const float inner = warp_sum(active ? grad_prob * prob : 0.0f);
  float grad_logit = 0.0f;
  if (active) {
    const float grad_score = (grad_prob - inner) / total;
    grad_logit = grad_score * score * (1.0f - score);
  }
  grad_s[slot] = grad_logit;
  idx_s[slot] = active ? expert : -1;
  __syncwarp();

  if (token >= num_tokens) return;
  const int base = warp_in_block * kWarp;
  const long row = static_cast<long>(token) * num_experts;
  for (int e = lane; e < num_experts; e += kWarp) {
    float value = 0.0f;
    for (int k = 0; k < topk; ++k) {
      if (idx_s[base + k] == e) {
        value += grad_s[base + k];
      }
    }
    grad_logits[row + e] = static_cast<scalar_t>(value);
  }
}

inline int blocks_for_tokens(int num_tokens) {
  const int tokens_per_block = kThreadsPerBlock / kWarp;
  return (num_tokens + tokens_per_block - 1) / tokens_per_block;
}

void check_inputs(const at::Tensor& logits, const at::Tensor& indices) {
  TORCH_CHECK(logits.is_cuda() && indices.is_cuda(), "logits and indices must be CUDA tensors");
  TORCH_CHECK(logits.dim() == 2, "logits must be [num_tokens, num_experts]");
  TORCH_CHECK(indices.dim() == 2, "indices must be [num_tokens, topk]");
  TORCH_CHECK(logits.scalar_type() == at::kFloat || logits.scalar_type() == at::kHalf ||
                  logits.scalar_type() == at::kBFloat16,
              "logits must be fp32, fp16, or bf16");
  TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");
  TORCH_CHECK(logits.is_contiguous() && indices.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(logits.device() == indices.device(),
              "logits and indices must be on the same CUDA device");
  TORCH_CHECK(logits.size(0) == indices.size(0), "logits and indices disagree on num_tokens");
  TORCH_CHECK(indices.size(1) >= 1 && indices.size(1) <= kWarp,
              "this kernel requires 1 <= topk <= 32");
}

}  // namespace

std::vector<at::Tensor> replay_fwd_dense(const at::Tensor& logits, const at::Tensor& indices,
                                         double scaling) {
  check_inputs(logits, indices);
  const int num_tokens = logits.size(0);
  const int num_experts = logits.size(1);
  const int topk = indices.size(1);

  const at::cuda::CUDAGuard guard(logits.device());
  auto routing_probs = at::empty({num_tokens, num_experts}, logits.options());
  auto routing_map = at::empty({num_tokens, num_experts}, logits.options().dtype(at::kBool));
  // Megatron promotes sigmoid and normalization to fp32 even for a low-precision router.
  auto scores = at::empty({num_tokens, topk}, logits.options().dtype(at::kFloat));
  auto denom = at::empty({num_tokens}, logits.options().dtype(at::kFloat));

  if (num_tokens > 0) {
    const int warps_per_block = kThreadsPerBlock / kWarp;
    const size_t smem = warps_per_block * kWarp * (sizeof(float) + sizeof(int));
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                    logits.scalar_type(),
                                    "replay_fwd_dense", [&] {
      replay_fwd_dense_kernel<scalar_t><<<blocks_for_tokens(num_tokens), kThreadsPerBlock, smem,
                                          at::cuda::getCurrentCUDAStream()>>>(
          logits.data_ptr<scalar_t>(), indices.data_ptr<int>(), routing_probs.data_ptr<scalar_t>(),
          routing_map.data_ptr<bool>(), scores.data_ptr<float>(), denom.data_ptr<float>(),
          num_tokens, num_experts, topk, static_cast<float>(scaling));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {routing_probs, routing_map, scores, denom};
}

at::Tensor replay_bwd_dense(const at::Tensor& grad_routing_probs, const at::Tensor& scores,
                            const at::Tensor& denom, const at::Tensor& indices,
                            int64_t num_experts, double scaling) {
  TORCH_CHECK(grad_routing_probs.is_cuda() && scores.is_cuda() && denom.is_cuda() &&
                  indices.is_cuda(),
              "backward inputs must be CUDA tensors");
  TORCH_CHECK(grad_routing_probs.scalar_type() == at::kFloat ||
                  grad_routing_probs.scalar_type() == at::kHalf ||
                  grad_routing_probs.scalar_type() == at::kBFloat16,
              "grad_routing_probs must be fp32, fp16, or bf16");
  TORCH_CHECK(scores.scalar_type() == at::kFloat && denom.scalar_type() == at::kFloat,
              "scores and denom must be fp32");
  TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");
  TORCH_CHECK(scores.is_contiguous() && denom.is_contiguous() && indices.is_contiguous(),
              "saved backward inputs must be contiguous");
  TORCH_CHECK(grad_routing_probs.device() == scores.device() && scores.device() == denom.device() &&
                  denom.device() == indices.device(),
              "backward inputs must be on the same CUDA device");
  const int num_tokens = scores.size(0);
  const int topk = indices.size(1);
  TORCH_CHECK(topk >= 1 && topk <= kWarp, "this kernel requires 1 <= topk <= 32");
  TORCH_CHECK(grad_routing_probs.dim() == 2 && grad_routing_probs.size(0) == num_tokens &&
                  grad_routing_probs.size(1) == num_experts,
              "grad_routing_probs shape does not match saved router shape");
  TORCH_CHECK(scores.dim() == 2 && indices.dim() == 2 && scores.sizes() == indices.sizes(),
              "scores and indices must have matching [num_tokens, topk] shapes");
  TORCH_CHECK(denom.dim() == 1 && denom.size(0) == num_tokens,
              "denom must have shape [num_tokens]");

  const at::cuda::CUDAGuard guard(scores.device());
  auto contiguous_grad = grad_routing_probs.contiguous();
  auto grad_logits = at::empty({num_tokens, num_experts}, grad_routing_probs.options());
  if (num_tokens > 0) {
    const int warps_per_block = kThreadsPerBlock / kWarp;
    const size_t smem = warps_per_block * kWarp * (sizeof(float) + sizeof(int));
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                    grad_routing_probs.scalar_type(),
                                    "replay_bwd_dense", [&] {
      replay_bwd_dense_kernel<scalar_t><<<blocks_for_tokens(num_tokens), kThreadsPerBlock, smem,
                                          at::cuda::getCurrentCUDAStream()>>>(
          contiguous_grad.data_ptr<scalar_t>(), scores.data_ptr<float>(), denom.data_ptr<float>(),
          indices.data_ptr<int>(), grad_logits.data_ptr<scalar_t>(), num_tokens,
          static_cast<int>(num_experts), topk, static_cast<float>(scaling));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return grad_logits;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("replay_fwd_dense", &replay_fwd_dense,
        "fused MoE router replay forward, Megatron dense contract");
  m.def("replay_bwd_dense", &replay_bwd_dense, "fused MoE router replay backward");
}
