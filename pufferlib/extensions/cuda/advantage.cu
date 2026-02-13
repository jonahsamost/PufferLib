#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/util/BFloat16.h>

namespace pufferlib {

// TIn = input type (bf16 or float), TOut = output type (always float for precision)
template<typename TIn, typename TOut>
__host__ __device__ void puff_advantage_row_cuda_fallback(
    const TIn* values, const TIn* rewards, const TIn* dones,
    const TIn* importance, TOut* advantages, float gamma, float lambda,
    float rho_clip, float c_clip, int horizon
) {
    float lastpufferlam = 0;
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0f - float(dones[t_next]);
        float imp = float(importance[t]);
        float rho_t = fminf(imp, rho_clip);
        float c_t = fminf(imp, c_clip);
        float delta = rho_t*(float(rewards[t_next]) + gamma*float(values[t_next])*nextnonterminal - float(values[t]));
        lastpufferlam = delta + gamma*lambda*c_t*lastpufferlam*nextnonterminal;
        advantages[t] = TOut(lastpufferlam);
    }
}

__device__ __forceinline__ void adv_vec_load(const float* ptr, float* out) {
    float4 v = *reinterpret_cast<const float4*>(ptr);
    out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
}

__device__ __forceinline__ void adv_vec_load(const c10::BFloat16* ptr, float* out) {
    uint4 raw = *reinterpret_cast<const uint4*>(ptr);
    const __nv_bfloat16* bf = reinterpret_cast<const __nv_bfloat16*>(&raw);
    #pragma unroll
    for (int i = 0; i < 8; i++) out[i] = __bfloat162float(bf[i]);
}

template<typename TIn, typename TOut>
__device__ __forceinline__ void puff_advantage_row_cuda(
    const TIn* values, const TIn* rewards, const TIn* dones,
    const TIn* importance, TOut* advantages, float gamma, float lambda,
    float rho_clip, float c_clip, int horizon
) {
    constexpr int N = 16 / sizeof(TIn);

    float lastpufferlam = 0.0f;
    int num_chunks = horizon / N;

    // Track values across chunk boundaries
    float next_value = float(values[horizon - 1]);
    float next_done = float(dones[horizon - 1]);
    float next_reward = float(rewards[horizon - 1]);

    // Process chunks from end to beginning
    for (int chunk = num_chunks - 1; chunk >= 0; chunk--) {
        int base = chunk * N;

        float v[N], r[N], d[N], imp[N];
        adv_vec_load(values + base, v);
        adv_vec_load(rewards + base, r);
        adv_vec_load(dones + base, d);
        adv_vec_load(importance + base, imp);

        float adv[N] = {0};
        // Last chunk: skip element N-1 (horizon-1 doesn't produce an advantage)
        int start_idx = (chunk == num_chunks - 1) ? (N - 2) : (N - 1);

        #pragma unroll
        for (int i = start_idx; i >= 0; i--) {
            float nextnonterminal = 1.0f - next_done;
            float rho_t = fminf(imp[i], rho_clip);
            float c_t = fminf(imp[i], c_clip);
            float delta = rho_t * (next_reward + gamma * next_value * nextnonterminal - v[i]);
            lastpufferlam = delta + gamma * lambda * c_t * lastpufferlam * nextnonterminal;
            adv[i] = lastpufferlam;
            next_value = v[i];
            next_done = d[i];
            next_reward = r[i];
        }

        *reinterpret_cast<float4*>(advantages + base) =
            make_float4(adv[0], adv[1], adv[2], adv[3]);
        if (N > 4) {
            *reinterpret_cast<float4*>(advantages + base + 4) =
                make_float4(adv[4], adv[5], adv[6], adv[7]);
        }
    }
}

void vtrace_check_cuda(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        int num_steps, int horizon) {

    // Validate input tensors
    torch::Device device = values.device();
    auto input_dtype = values.dtype();
    for (const torch::Tensor& t : {values, rewards, dones, importance}) {
        TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
        TORCH_CHECK(t.dtype() == input_dtype, "Input tensors must have matching dtype");
        if (!t.is_contiguous()) {
            t.contiguous();
        }
    }
    // advantages can be different dtype (fp32 for precision)
    TORCH_CHECK(advantages.dim() == 2, "Advantages must be 2D");
    TORCH_CHECK(advantages.device() == device, "Advantages must be on same device");
    TORCH_CHECK(advantages.size(0) == num_steps, "Advantages first dimension must match");
    TORCH_CHECK(advantages.size(1) == horizon, "Advantages second dimension must match");
    if (!advantages.is_contiguous()) {
        advantages.contiguous();
    }
}

template<typename TIn, typename TOut>
__global__ void puff_advantage_kernel(const TIn* values, const TIn* rewards,
        const TIn* dones, const TIn* importance, TOut* advantages, float gamma,
        float lambda, float rho_clip, float c_clip, int num_steps, int horizon) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= num_steps) return;
    int offset = row*horizon;
    puff_advantage_row_cuda<TIn, TOut>(values + offset, rewards + offset, dones + offset,
        importance + offset, advantages + offset, gamma, lambda, rho_clip, c_clip, horizon);
}

// Scalar kernel (fallback for unaligned horizons)
template<typename TIn, typename TOut>
__global__ void puff_advantage_kernel_scalar(const TIn* values, const TIn* rewards,
        const TIn* dones, const TIn* importance, TOut* advantages, float gamma,
        float lambda, float rho_clip, float c_clip, int num_steps, int horizon) {
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row >= num_steps) return;
    int offset = row*horizon;
    puff_advantage_row_cuda_fallback<TIn, TOut>(values + offset, rewards + offset, dones + offset,
        importance + offset, advantages + offset, gamma, lambda, rho_clip, c_clip, horizon);
}

template<typename TIn, typename TOut>
void compute_puff_advantage_cuda_impl(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    vtrace_check_cuda(values, rewards, dones, importance, advantages, num_steps, horizon);
    TORCH_CHECK(values.is_cuda(), "All tensors must be on GPU");

    int threads_per_block = 256;
    int blocks = (num_steps + threads_per_block - 1) / threads_per_block;

    constexpr int N = 16 / sizeof(TIn);
    // auto kernel = (horizon % N == 0 && sizeof(TOut) == 4)
    //     ? puff_advantage_kernel<TIn, TOut>
    //     : puff_advantage_kernel_scalar<TIn, TOut>;
    auto kernel = puff_advantage_kernel_scalar<TIn, TOut>;

    kernel<<<blocks, threads_per_block>>>(
        values.data_ptr<TIn>(), rewards.data_ptr<TIn>(),
        dones.data_ptr<TIn>(), importance.data_ptr<TIn>(),
        advantages.data_ptr<TOut>(),
        static_cast<float>(gamma), static_cast<float>(lambda),
        static_cast<float>(rho_clip), static_cast<float>(c_clip),
        num_steps, horizon);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void compute_puff_advantage_cuda(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip) {
    auto input_dtype = values.dtype();
    auto output_dtype = advantages.dtype();
    
    // Support bf16 inputs with fp32 output for precision
    if (input_dtype == torch::kFloat32 && output_dtype == torch::kFloat32) {
        compute_puff_advantage_cuda_impl<float, float>(values, rewards, dones, importance, advantages,
            gamma, lambda, rho_clip, c_clip);
    } else if (input_dtype == torch::kBFloat16 && output_dtype == torch::kFloat32) {
        compute_puff_advantage_cuda_impl<at::BFloat16, float>(values, rewards, dones, importance, advantages,
            gamma, lambda, rho_clip, c_clip);
    } else if (input_dtype == torch::kBFloat16 && output_dtype == torch::kBFloat16) {
        compute_puff_advantage_cuda_impl<at::BFloat16, at::BFloat16>(values, rewards, dones, importance, advantages,
            gamma, lambda, rho_clip, c_clip);
    } else {
        TORCH_CHECK(false, "Unsupported dtype combination: inputs must be float32 or bfloat16, advantages must be float32 or bfloat16");
    }
}

// ============================================================================
// Scalar-only dispatch (for benchmarking, can be deleted when not needed)
// ============================================================================

template<typename TIn, typename TOut>
void compute_puff_advantage_cuda_scalar_impl(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    vtrace_check_cuda(values, rewards, dones, importance, advantages, num_steps, horizon);
    TORCH_CHECK(values.is_cuda(), "All tensors must be on GPU");

    int threads_per_block = 256;
    int blocks = (num_steps + threads_per_block - 1) / threads_per_block;

    puff_advantage_kernel_scalar<TIn, TOut><<<blocks, threads_per_block>>>(
        values.data_ptr<TIn>(),
        rewards.data_ptr<TIn>(),
        dones.data_ptr<TIn>(),
        importance.data_ptr<TIn>(),
        advantages.data_ptr<TOut>(),
        static_cast<float>(gamma),
        static_cast<float>(lambda),
        static_cast<float>(rho_clip),
        static_cast<float>(c_clip),
        num_steps,
        horizon
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void compute_puff_advantage_cuda_scalar(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip) {
    auto input_dtype = values.dtype();
    auto output_dtype = advantages.dtype();
    
    if (input_dtype == torch::kFloat32 && output_dtype == torch::kFloat32) {
        compute_puff_advantage_cuda_scalar_impl<float, float>(values, rewards, dones, importance, advantages,
            gamma, lambda, rho_clip, c_clip);
    } else if (input_dtype == torch::kBFloat16 && output_dtype == torch::kFloat32) {
        compute_puff_advantage_cuda_scalar_impl<at::BFloat16, float>(values, rewards, dones, importance, advantages,
            gamma, lambda, rho_clip, c_clip);
    } else if (input_dtype == torch::kBFloat16 && output_dtype == torch::kBFloat16) {
        compute_puff_advantage_cuda_scalar_impl<at::BFloat16, at::BFloat16>(values, rewards, dones, importance, advantages,
            gamma, lambda, rho_clip, c_clip);
    } else {
        TORCH_CHECK(false, "Unsupported dtype combination");
    }
}

TORCH_LIBRARY_IMPL(pufferlib, CUDA, m) {
  m.impl("compute_puff_advantage", &compute_puff_advantage_cuda);
}

}
