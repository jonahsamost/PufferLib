// NMMO3 CUDA encoder: multihot, cuDNN conv, embedding, concat, projection
// Included by pufferlib.cu — requires precision_t, PrecisionTensor, Allocator, puf_mm, etc.

#include "cudnn_conv2d.cu"

// ---- NMMO3 constants ----

static constexpr int N3_MAP_H = 11, N3_MAP_W = 15, N3_NFEAT = 10;
static constexpr int N3_MULTIHOT = 59;
static constexpr int N3_MAP_SIZE = N3_MAP_H * N3_MAP_W * N3_NFEAT;
static constexpr int N3_PLAYER = 47, N3_REWARD = 10;
static constexpr int N3_EMBED_DIM = 32, N3_EMBED_VOCAB = 128;
static constexpr int N3_PLAYER_EMBED = N3_PLAYER * N3_EMBED_DIM;
static constexpr int N3_C1_IC = 59, N3_C1_OC = 128, N3_C1_K = 5, N3_C1_S = 3;
static constexpr int N3_C1_OH = 3, N3_C1_OW = 4;
static constexpr int N3_C2_IC = 128, N3_C2_OC = 128, N3_C2_K = 3, N3_C2_S = 1;
static constexpr int N3_C2_OH = 1, N3_C2_OW = 2;
static constexpr int N3_CONV_FLAT = N3_C2_OC * N3_C2_OH * N3_C2_OW;
static constexpr int N3_CONCAT = N3_CONV_FLAT + N3_PLAYER_EMBED + N3_PLAYER + N3_REWARD;

__constant__ int N3_OFFSETS[10] = {0, 4, 8, 25, 30, 33, 38, 43, 48, 55};

static cudnnDataType_t n3_cudnn_dtype() {
    return (PRECISION_SIZE == 2) ? CUDNN_DATA_BFLOAT16 : CUDNN_DATA_FLOAT;
}

// ---- NMMO3 kernels ----

__global__ void n3_multihot_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ obs, int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_MAP_H * N3_MAP_W) return;
    int b = idx / (N3_MAP_H * N3_MAP_W), rem = idx % (N3_MAP_H * N3_MAP_W);
    int h = rem / N3_MAP_W, w = rem % N3_MAP_W;
    const precision_t* src = obs + b * obs_size + (h * N3_MAP_W + w) * N3_NFEAT;
    precision_t* dst = out + b * N3_MULTIHOT * N3_MAP_H * N3_MAP_W;
    for (int f = 0; f < N3_NFEAT; f++)
        dst[(N3_OFFSETS[f] + (int)to_float(src[f])) * N3_MAP_H * N3_MAP_W + h * N3_MAP_W + w] = from_float(1.0f);
}

__global__ void n3_embedding_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ obs,
    const precision_t* __restrict__ embed_w, int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_PLAYER) return;
    int b = idx / N3_PLAYER, f = idx % N3_PLAYER;
    int val = (int)to_float(obs[b * obs_size + N3_MAP_SIZE + f]);
    const precision_t* src = embed_w + val * N3_EMBED_DIM;
    precision_t* dst = out + b * N3_PLAYER_EMBED + f * N3_EMBED_DIM;
    for (int d = 0; d < N3_EMBED_DIM; d++) dst[d] = src[d];
}

__global__ void n3_concat_kernel(
    precision_t* __restrict__ out, const precision_t* __restrict__ conv_flat,
    const precision_t* __restrict__ embed, const precision_t* __restrict__ obs,
    int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_CONCAT) return;
    int b = idx / N3_CONCAT, c = idx % N3_CONCAT;
    precision_t val;
    if (c < N3_CONV_FLAT) {
        int oc = c / (N3_C2_OH * N3_C2_OW), r = c % (N3_C2_OH * N3_C2_OW);
        int oh = r / N3_C2_OW, ow = r % N3_C2_OW;
        val = conv_flat[b * N3_CONV_FLAT + oc * N3_C2_OH * N3_C2_OW + oh * N3_C2_OW + ow];
    } else if (c < N3_CONV_FLAT + N3_PLAYER_EMBED)
        val = embed[b * N3_PLAYER_EMBED + (c - N3_CONV_FLAT)];
    else if (c < N3_CONV_FLAT + N3_PLAYER_EMBED + N3_PLAYER)
        val = obs[b * obs_size + N3_MAP_SIZE + (c - N3_CONV_FLAT - N3_PLAYER_EMBED)];
    else
        val = obs[b * obs_size + obs_size - N3_REWARD + (c - N3_CONV_FLAT - N3_PLAYER_EMBED - N3_PLAYER)];
    out[idx] = val;
}

__global__ void n3_bias_relu_kernel(
    precision_t* __restrict__ data, const precision_t* __restrict__ bias, int total, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    data[idx] = from_float(fmaxf(0.0f, to_float(data[idx]) + to_float(bias[idx % dim])));
}

__global__ void n3_relu_backward_kernel(
    precision_t* __restrict__ grad, const precision_t* __restrict__ out, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    if (to_float(out[idx]) <= 0.0f) grad[idx] = from_float(0.0f);
}


__global__ void bias_grad_kernel(
    precision_t* __restrict__ bgrad, const precision_t* __restrict__ grad, int N, int dim) {
    int d = blockIdx.x;
    if (d >= dim) return;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        sum += to_float(grad[i * dim + d]);
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float sdata[32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) sdata[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = (lane < (blockDim.x + 31) / 32) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) bgrad[d] = from_float(sum);
    }
}

// NCHW bias grad: sum over (B, OH, OW) for each OC channel
__global__ void n3_conv_bias_grad_nchw(
    precision_t* __restrict__ bgrad, const precision_t* __restrict__ grad,
    int B, int OC, int spatial) {
    int oc = blockIdx.x;
    if (oc >= OC) return;
    float sum = 0.0f;
    int total = B * spatial;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int b = i / spatial, s = i % spatial;
        sum += to_float(grad[b * OC * spatial + oc * spatial + s]);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float sdata[32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) sdata[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = (lane < (blockDim.x + 31) / 32) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) bgrad[oc] = from_float(sum);
    }
}

__global__ void n3_concat_backward_conv_kernel(
    precision_t* __restrict__ conv_grad, const precision_t* __restrict__ concat_grad, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_CONV_FLAT) return;
    int b = idx / N3_CONV_FLAT, c = idx % N3_CONV_FLAT;
    conv_grad[b * N3_CONV_FLAT + c] = concat_grad[b * N3_CONCAT + c];
}

// Embedding backward: scatter-add grad from concat_grad's player_embed region
// into embed_wgrad (float accumulation buffer).
// Each (b, f) looked up row obs[b, MAP_SIZE+f] from the table.
__global__ void n3_embedding_backward_kernel(
    float* __restrict__ embed_wgrad_f,
    const precision_t* __restrict__ concat_grad,
    const precision_t* __restrict__ obs,
    int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N3_PLAYER * N3_EMBED_DIM) return;
    int b = idx / (N3_PLAYER * N3_EMBED_DIM);
    int rem = idx % (N3_PLAYER * N3_EMBED_DIM);
    int f = rem / N3_EMBED_DIM;
    int d = rem % N3_EMBED_DIM;
    int val = (int)to_float(obs[b * obs_size + N3_MAP_SIZE + f]);
    float g = to_float(concat_grad[b * N3_CONCAT + N3_CONV_FLAT + f * N3_EMBED_DIM + d]);
    atomicAdd(&embed_wgrad_f[val * N3_EMBED_DIM + d], g);
}

// Cast float buffer to precision_t
__global__ void n3_float_to_precision_kernel(
    precision_t* __restrict__ dst, const float* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = from_float(src[idx]);
}

// ---- atomicAdd for precision_t ----
#ifdef PRECISION_FLOAT
__device__ __forceinline__ void atomicAdd_precision(precision_t* addr, precision_t val) {
    atomicAdd(addr, val);
}
#else
__device__ __forceinline__ void atomicAdd_precision(precision_t* addr, precision_t val) {
    // bf16 atomicAdd via CAS on enclosing 32-bit word
    unsigned int* addr_u32 = (unsigned int*)((size_t)addr & ~2ULL);
    bool is_high = ((size_t)addr & 2) != 0;
    unsigned int old_u32 = *addr_u32, assumed;
    do {
        assumed = old_u32;
        __nv_bfloat16* pair = (__nv_bfloat16*)&old_u32;
        float sum = __bfloat162float(pair[is_high]) + __bfloat162float(val);
        unsigned int new_u32 = assumed;
        ((__nv_bfloat16*)&new_u32)[is_high] = __float2bfloat16(sum);
        old_u32 = atomicCAS(addr_u32, assumed, new_u32);
    } while (old_u32 != assumed);
}
#endif

// ---- NCHW bias kernels for im2col conv path ----

__global__ void conv_bias_kernel(precision_t* __restrict__ data,
        const precision_t* __restrict__ bias, int B, int OC, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OC * spatial;
    if (idx >= total) return;
    int oc = (idx / spatial) % OC;
    data[idx] = from_float(to_float(data[idx]) + to_float(bias[oc]));
}

__global__ void conv_bias_relu_kernel(precision_t* __restrict__ data,
        const precision_t* __restrict__ bias, int B, int OC, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OC * spatial;
    if (idx >= total) return;
    int oc = (idx / spatial) % OC;
    data[idx] = from_float(fmaxf(0.0f, to_float(data[idx]) + to_float(bias[oc])));
}

// ---- im2col + cuBLAS conv (no cuDNN) ----
// NCHW layout throughout. Weight stored as (OC, IC*K*K).
// im2col produces (B*OH*OW, IC*K*K), matmul with W^T gives (B*OH*OW, OC),
// then reshape to NCHW (B, OC, OH, OW).

__global__ void im2col_kernel(
    const precision_t* __restrict__ input, precision_t* __restrict__ col,
    int B, int IC, int IH, int IW, int K, int S, int OH, int OW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OH * OW * IC * K * K;
    if (idx >= total) return;
    int col_w = IC * K * K;
    int row = idx / col_w;
    int c = idx % col_w;
    int b = row / (OH * OW);
    int rem = row % (OH * OW);
    int oh = rem / OW, ow = rem % OW;
    int ic = c / (K * K), kk = c % (K * K);
    int kh = kk / K, kw = kk % K;
    int ih = oh * S + kh, iw = ow * S + kw;
    col[idx] = input[b * IC * IH * IW + ic * IH * IW + ih * IW + iw];
}

// Backward: col2im — input-centric gather to avoid atomics.
// Each thread owns one (b, ic, ih, iw) element and sums contributions from all
// (oh, ow, kh, kw) patches that map to it.
__global__ void col2im_kernel(
    const precision_t* __restrict__ col, precision_t* __restrict__ grad_input,
    int B, int IC, int IH, int IW, int K, int S, int OH, int OW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * IC * IH * IW;
    if (idx >= total) return;
    int iw = idx % IW;
    int ih = (idx / IW) % IH;
    int ic = (idx / (IW * IH)) % IC;
    int b  = idx / (IW * IH * IC);
    float sum = 0.0f;
    for (int kh = 0; kh < K; kh++) {
        int ih_off = ih - kh;
        if (ih_off < 0 || ih_off % S != 0) continue;
        int oh = ih_off / S;
        if (oh >= OH) continue;
        for (int kw = 0; kw < K; kw++) {
            int iw_off = iw - kw;
            if (iw_off < 0 || iw_off % S != 0) continue;
            int ow = iw_off / S;
            if (ow >= OW) continue;
            int col_idx = (b * OH * OW + oh * OW + ow) * (IC * K * K) + ic * K * K + kh * K + kw;
            sum += to_float(col[col_idx]);
        }
    }
    grad_input[idx] = from_float(sum);
}

// Transpose (B, OC, OH, OW) -> (B*OH*OW, OC)  [NCHW to row-major spatial-first]
__global__ void nchw_to_rows_kernel(
    const precision_t* __restrict__ src, precision_t* __restrict__ dst,
    int B, int OC, int spatial
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OC * spatial;
    if (idx >= total) return;
    int b = idx / (OC * spatial);
    int oc = (idx / spatial) % OC;
    int s = idx % spatial;
    dst[(b * spatial + s) * OC + oc] = src[idx];
}

// Transpose (B*OH*OW, OC) -> (B, OC, OH, OW)  [row-major spatial-first to NCHW]
__global__ void rows_to_nchw_kernel(
    const precision_t* __restrict__ src, precision_t* __restrict__ dst,
    int B, int OC, int spatial
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OC * spatial;
    if (idx >= total) return;
    int b = idx / (OC * spatial);
    int oc = (idx / spatial) % OC;
    int s = idx % spatial;
    dst[idx] = src[(b * spatial + s) * OC + oc];
}

// Forward: im2col conv + bias + optional relu. All NCHW.
// col_buf: pre-allocated (max_B * OH * OW, IC * K * K)
// mm_buf:  pre-allocated (max_B * OH * OW, OC)  — row-major (spatial-first)
static void gemm_conv_forward(
    PrecisionTensor* weight, PrecisionTensor* bias,
    precision_t* input, precision_t* output,
    precision_t* col_buf, precision_t* mm_buf,
    int B, int IC, int IH, int IW, int OC, int K, int S, int OH, int OW,
    bool relu, cudaStream_t stream
) {
    int col_rows = B * OH * OW;
    int col_cols = IC * K * K;
    int total_col = col_rows * col_cols;
    int total_out = B * OC * OH * OW;

    // im2col: input NCHW -> col (B*OH*OW, IC*K*K)
    im2col_kernel<<<grid_size(total_col), BLOCK_SIZE, 0, stream>>>(
        input, col_buf, B, IC, IH, IW, K, S, OH, OW);

    // matmul: col (B*OH*OW, IC*K*K) @ W^T (IC*K*K, OC) = mm_buf (B*OH*OW, OC)
    PrecisionTensor col_t = {.data = col_buf, .shape = {col_rows, col_cols}};
    PrecisionTensor mm_t  = {.data = mm_buf,  .shape = {col_rows, OC}};
    puf_mm(&col_t, weight, &mm_t, stream);

    // transpose (B*OH*OW, OC) -> (B, OC, OH, OW) NCHW + bias + relu
    int spatial = OH * OW;
    rows_to_nchw_kernel<<<grid_size(total_out), BLOCK_SIZE, 0, stream>>>(
        mm_buf, output, B, OC, spatial);
    if (relu) {
        conv_bias_relu_kernel<<<grid_size(total_out), BLOCK_SIZE, 0, stream>>>(
            output, bias->data, B, OC, spatial);
    } else {
        conv_bias_kernel<<<grid_size(total_out), BLOCK_SIZE, 0, stream>>>(
            output, bias->data, B, OC, spatial);
    }
}

// Backward: weight grad + optional input grad via im2col/col2im + cuBLAS.
// grad_output is NCHW (B, OC, OH, OW). saved_input is NCHW.
// Caller handles relu backward and bias grad (same as cuDNN path).
static void gemm_conv_backward(
    PrecisionTensor* weight,
    precision_t* saved_input, precision_t* grad_output,
    precision_t* wgrad, precision_t* input_grad,
    precision_t* col_buf, precision_t* mm_buf,
    int B, int IC, int IH, int IW, int OC, int K, int S, int OH, int OW,
    cudaStream_t stream
) {
    int col_rows = B * OH * OW;
    int col_cols = IC * K * K;
    int total_col = col_rows * col_cols;
    int total_out = B * OC * OH * OW;
    int spatial = OH * OW;

    // Transpose grad_output NCHW -> (B*OH*OW, OC)
    nchw_to_rows_kernel<<<grid_size(total_out), BLOCK_SIZE, 0, stream>>>(
        grad_output, mm_buf, B, OC, spatial);

    // im2col of saved_input
    im2col_kernel<<<grid_size(total_col), BLOCK_SIZE, 0, stream>>>(
        saved_input, col_buf, B, IC, IH, IW, K, S, OH, OW);

    // Weight grad: mm_buf^T (OC, B*OH*OW) @ col_buf (B*OH*OW, IC*K*K) = wgrad (OC, IC*K*K)
    PrecisionTensor mm_t  = {.data = mm_buf,  .shape = {col_rows, OC}};
    PrecisionTensor col_t = {.data = col_buf, .shape = {col_rows, col_cols}};
    PrecisionTensor wg_t  = {.data = wgrad,   .shape = {OC, col_cols}};
    puf_mm_tn(&mm_t, &col_t, &wg_t, stream);

    // Input grad (optional): mm_buf (B*OH*OW, OC) @ weight (OC, IC*K*K) = col_grad (B*OH*OW, IC*K*K)
    if (input_grad) {
        puf_mm_nn(&mm_t, weight, &col_t, stream);  // reuse col_buf as col_grad
        col2im_kernel<<<grid_size(B * IC * IH * IW), BLOCK_SIZE, 0, stream>>>(
            col_buf, input_grad, B, IC, IH, IW, K, S, OH, OW);
    }
}

// ---- NMMO3 encoder structs ----

struct NMMO3EncoderWeights {
    ConvWeights conv1, conv2;
    PrecisionTensor embed_w, proj_w, proj_b;
    int obs_size, hidden;
};

struct NMMO3EncoderActivations {
    ConvActivations conv1, conv2;
    PrecisionTensor col1, mm1, col2, mm2;  // im2col + matmul scratch buffers
    PrecisionTensor multihot, embed_out, concat, out, saved_obs;
    PrecisionTensor embed_wgrad, proj_wgrad, proj_bgrad;
    FloatTensor embed_wgrad_f;  // float accumulation buffer for scatter-add
};

static NMMO3EncoderWeights* nmmo3_encoder_create(int obs_size, int hidden) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)calloc(1, sizeof(NMMO3EncoderWeights));
    ew->obs_size = obs_size; ew->hidden = hidden;
    conv_init(&ew->conv1, N3_C1_IC, N3_C1_OC, N3_C1_K, N3_C1_S, N3_MAP_H, N3_MAP_W, true);
    conv_init(&ew->conv2, N3_C2_IC, N3_C2_OC, N3_C2_K, N3_C2_S, N3_C1_OH, N3_C1_OW, false);
    return ew;
}

// ---- NMMO3 encoder interface ----

static PrecisionTensor nmmo3_encoder_forward(void* w, void* activations, PrecisionTensor input, cudaStream_t stream) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    NMMO3EncoderActivations* a = (NMMO3EncoderActivations*)activations;
    int B = input.shape[0];

    if (a->saved_obs.data) puf_copy(&a->saved_obs, &input, stream);

    cudaMemsetAsync(a->multihot.data, 0, (int64_t)B * N3_MULTIHOT * N3_MAP_H * N3_MAP_W * sizeof(precision_t), stream);
    n3_multihot_kernel<<<grid_size(B * N3_MAP_H * N3_MAP_W), BLOCK_SIZE, 0, stream>>>(
        a->multihot.data, input.data, B, ew->obs_size);

    gemm_conv_forward(&ew->conv1.w, &ew->conv1.b, a->multihot.data, a->conv1.out.data,
        a->col1.data, a->mm1.data, B, N3_C1_IC, N3_MAP_H, N3_MAP_W,
        N3_C1_OC, N3_C1_K, N3_C1_S, N3_C1_OH, N3_C1_OW, true, stream);
    if (a->conv1.saved_input.data)
        cudaMemcpyAsync(a->conv1.saved_input.data, a->multihot.data,
            (int64_t)B * N3_C1_IC * N3_MAP_H * N3_MAP_W * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream);
    gemm_conv_forward(&ew->conv2.w, &ew->conv2.b, a->conv1.out.data, a->conv2.out.data,
        a->col2.data, a->mm2.data, B, N3_C2_IC, N3_C1_OH, N3_C1_OW,
        N3_C2_OC, N3_C2_K, N3_C2_S, N3_C2_OH, N3_C2_OW, false, stream);
    if (a->conv2.saved_input.data)
        cudaMemcpyAsync(a->conv2.saved_input.data, a->conv1.out.data,
            (int64_t)B * N3_C2_IC * N3_C1_OH * N3_C1_OW * sizeof(precision_t), cudaMemcpyDeviceToDevice, stream);

    n3_embedding_kernel<<<grid_size(B * N3_PLAYER), BLOCK_SIZE, 0, stream>>>(
        a->embed_out.data, input.data, ew->embed_w.data, B, ew->obs_size);
    n3_concat_kernel<<<grid_size(B * N3_CONCAT), BLOCK_SIZE, 0, stream>>>(
        a->concat.data, a->conv2.out.data, a->embed_out.data, input.data, B, ew->obs_size);

    puf_mm(&a->concat, &ew->proj_w, &a->out, stream);
    n3_bias_relu_kernel<<<grid_size(B * ew->hidden), BLOCK_SIZE, 0, stream>>>(
        a->out.data, ew->proj_b.data, B * ew->hidden, ew->hidden);
    return a->out;
}

static void nmmo3_encoder_backward(void* w, void* activations, PrecisionTensor grad, cudaStream_t stream) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    NMMO3EncoderActivations* a = (NMMO3EncoderActivations*)activations;
    int B = grad.shape[0], H = ew->hidden;

    n3_relu_backward_kernel<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
        grad.data, a->out.data, B * H);
    bias_grad_kernel<<<H, 256, 0, stream>>>(
        a->proj_bgrad.data, grad.data, B, H);
    puf_mm_tn(&grad, &a->concat, &a->proj_wgrad, stream);

    PrecisionTensor grad_concat = {.data = a->concat.data, .shape = {B, N3_CONCAT}};
    puf_mm_nn(&grad, &ew->proj_w, &grad_concat, stream);

    n3_concat_backward_conv_kernel<<<grid_size(B * N3_CONV_FLAT), BLOCK_SIZE, 0, stream>>>(
        a->conv2.grad.data, grad_concat.data, B);

    n3_conv_bias_grad_nchw<<<ew->conv2.OC, 256, 0, stream>>>(
        a->conv2.bgrad.data, a->conv2.grad.data,
        B, ew->conv2.OC, ew->conv2.OH * ew->conv2.OW);
    gemm_conv_backward(&ew->conv2.w, a->conv2.saved_input.data, a->conv2.grad.data,
        a->conv2.wgrad.data, a->conv1.grad.data,
        a->col2.data, a->mm2.data, B, N3_C2_IC, N3_C1_OH, N3_C1_OW,
        N3_C2_OC, N3_C2_K, N3_C2_S, N3_C2_OH, N3_C2_OW, stream);

    n3_relu_backward_kernel<<<grid_size(B * ew->conv1.OC * ew->conv1.OH * ew->conv1.OW), BLOCK_SIZE, 0, stream>>>(
        a->conv1.grad.data, a->conv1.out.data,
        B * ew->conv1.OC * ew->conv1.OH * ew->conv1.OW);
    n3_conv_bias_grad_nchw<<<ew->conv1.OC, 256, 0, stream>>>(
        a->conv1.bgrad.data, a->conv1.grad.data,
        B, ew->conv1.OC, ew->conv1.OH * ew->conv1.OW);
    gemm_conv_backward(&ew->conv1.w, a->conv1.saved_input.data, a->conv1.grad.data,
        a->conv1.wgrad.data, NULL,
        a->col1.data, a->mm1.data, B, N3_C1_IC, N3_MAP_H, N3_MAP_W,
        N3_C1_OC, N3_C1_K, N3_C1_S, N3_C1_OH, N3_C1_OW, stream);

    // Embedding backward: scatter-add from concat gradient into float buffer, then cast
    int embed_n = N3_EMBED_VOCAB * N3_EMBED_DIM;
    cudaMemsetAsync(a->embed_wgrad_f.data, 0, embed_n * sizeof(float), stream);
    n3_embedding_backward_kernel<<<grid_size(B * N3_PLAYER * N3_EMBED_DIM), BLOCK_SIZE, 0, stream>>>(
        a->embed_wgrad_f.data, grad_concat.data, a->saved_obs.data, B, ew->obs_size);
    n3_float_to_precision_kernel<<<grid_size(embed_n), BLOCK_SIZE, 0, stream>>>(
        a->embed_wgrad.data, a->embed_wgrad_f.data, embed_n);
}

static void nmmo3_encoder_init_weights(void* w, uint64_t* seed, cudaStream_t stream) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    conv_init_weights(&ew->conv1, seed, stream);
    conv_init_weights(&ew->conv2, seed, stream);
    auto init2d = [&](PrecisionTensor& t, int rows, int cols, float gain) {
        PrecisionTensor wt = {.data = t.data, .shape = {rows, cols}};
        puf_kaiming_init(&wt, gain, (*seed)++, stream);
    };
    puf_normal_init(&ew->embed_w, 1.0f, (*seed)++, stream);
    init2d(ew->proj_w, ew->hidden, N3_CONCAT, 1.0f);
    cudaMemsetAsync(ew->proj_b.data, 0, numel(ew->proj_b.shape) * sizeof(precision_t), stream);
}

static void nmmo3_encoder_reg_params(void* w, Allocator* alloc) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    conv_reg_params(&ew->conv1, alloc);
    conv_reg_params(&ew->conv2, alloc);
    ew->embed_w = {.shape = {N3_EMBED_VOCAB, N3_EMBED_DIM}};
    ew->proj_w  = {.shape = {ew->hidden, N3_CONCAT}};
    ew->proj_b  = {.shape = {ew->hidden}};
    alloc_register(alloc,&ew->embed_w);
    alloc_register(alloc,&ew->proj_w);  alloc_register(alloc,&ew->proj_b);
}

static void nmmo3_encoder_reg_train(void* w, void* activations, Allocator* acts, Allocator* grads, int B_TT) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    NMMO3EncoderActivations* a = (NMMO3EncoderActivations*)activations;
    *a = {};
    a->multihot = {.shape = {B_TT, N3_MULTIHOT * N3_MAP_H * N3_MAP_W}};
    alloc_register(acts,&a->multihot);
    // Conv1 buffers
    a->conv1.out         = {.shape = {B_TT * N3_C1_OC * N3_C1_OH * N3_C1_OW}};
    a->conv1.grad        = {.shape = {B_TT * N3_C1_OC * N3_C1_OH * N3_C1_OW}};
    a->conv1.saved_input = {.shape = {B_TT * N3_C1_IC * N3_MAP_H * N3_MAP_W}};
    a->conv1.wgrad       = {.shape = {N3_C1_OC, N3_C1_IC * N3_C1_K * N3_C1_K}};
    a->conv1.bgrad       = {.shape = {N3_C1_OC}};
    alloc_register(acts,&a->conv1.out); alloc_register(acts,&a->conv1.grad); alloc_register(acts,&a->conv1.saved_input);
    alloc_register(grads,&a->conv1.wgrad); alloc_register(grads,&a->conv1.bgrad);
    a->col1 = {.shape = {B_TT * N3_C1_OH * N3_C1_OW, N3_C1_IC * N3_C1_K * N3_C1_K}};
    a->mm1  = {.shape = {B_TT * N3_C1_OH * N3_C1_OW, N3_C1_OC}};
    alloc_register(acts,&a->col1); alloc_register(acts,&a->mm1);
    // Conv2 buffers
    a->conv2.out         = {.shape = {B_TT * N3_C2_OC * N3_C2_OH * N3_C2_OW}};
    a->conv2.grad        = {.shape = {B_TT * N3_C2_OC * N3_C2_OH * N3_C2_OW}};
    a->conv2.saved_input = {.shape = {B_TT * N3_C2_IC * N3_C1_OH * N3_C1_OW}};
    a->conv2.wgrad       = {.shape = {N3_C2_OC, N3_C2_IC * N3_C2_K * N3_C2_K}};
    a->conv2.bgrad       = {.shape = {N3_C2_OC}};
    alloc_register(acts,&a->conv2.out); alloc_register(acts,&a->conv2.grad); alloc_register(acts,&a->conv2.saved_input);
    alloc_register(grads,&a->conv2.wgrad); alloc_register(grads,&a->conv2.bgrad);
    a->col2 = {.shape = {B_TT * N3_C2_OH * N3_C2_OW, N3_C2_IC * N3_C2_K * N3_C2_K}};
    a->mm2  = {.shape = {B_TT * N3_C2_OH * N3_C2_OW, N3_C2_OC}};
    alloc_register(acts,&a->col2); alloc_register(acts,&a->mm2);
    a->embed_out = {.shape = {B_TT, N3_PLAYER_EMBED}};
    a->concat    = {.shape = {B_TT, N3_CONCAT}};
    a->out       = {.shape = {B_TT, ew->hidden}};
    a->saved_obs = {.shape = {B_TT, ew->obs_size}};
    alloc_register(acts,&a->embed_out); alloc_register(acts,&a->concat);
    alloc_register(acts,&a->out);       alloc_register(acts,&a->saved_obs);
    a->embed_wgrad = {.shape = {N3_EMBED_VOCAB, N3_EMBED_DIM}};
    a->embed_wgrad_f = {.shape = {N3_EMBED_VOCAB, N3_EMBED_DIM}};
    a->proj_wgrad  = {.shape = {ew->hidden, N3_CONCAT}};
    a->proj_bgrad  = {.shape = {ew->hidden}};
    alloc_register(grads,&a->embed_wgrad);
    alloc_register(acts,&a->embed_wgrad_f);
    alloc_register(grads,&a->proj_wgrad);  alloc_register(grads,&a->proj_bgrad);
}

static void nmmo3_encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    NMMO3EncoderWeights* ew = (NMMO3EncoderWeights*)w;
    NMMO3EncoderActivations* a = (NMMO3EncoderActivations*)activations;
    a->multihot = {.shape = {B, N3_MULTIHOT * N3_MAP_H * N3_MAP_W}};
    alloc_register(alloc,&a->multihot);
    a->conv1.out = {.shape = {B * N3_C1_OC * N3_C1_OH * N3_C1_OW}};
    alloc_register(alloc,&a->conv1.out);
    a->col1 = {.shape = {B * N3_C1_OH * N3_C1_OW, N3_C1_IC * N3_C1_K * N3_C1_K}};
    a->mm1  = {.shape = {B * N3_C1_OH * N3_C1_OW, N3_C1_OC}};
    alloc_register(alloc,&a->col1); alloc_register(alloc,&a->mm1);
    a->conv2.out = {.shape = {B * N3_C2_OC * N3_C2_OH * N3_C2_OW}};
    alloc_register(alloc,&a->conv2.out);
    a->col2 = {.shape = {B * N3_C2_OH * N3_C2_OW, N3_C2_IC * N3_C2_K * N3_C2_K}};
    a->mm2  = {.shape = {B * N3_C2_OH * N3_C2_OW, N3_C2_OC}};
    alloc_register(alloc,&a->col2); alloc_register(alloc,&a->mm2);
    a->embed_out = {.shape = {B, N3_PLAYER_EMBED}};
    a->concat    = {.shape = {B, N3_CONCAT}};
    a->out       = {.shape = {B, ew->hidden}};
    alloc_register(alloc,&a->embed_out); alloc_register(alloc,&a->concat); alloc_register(alloc,&a->out);
}

static void* nmmo3_encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    return nmmo3_encoder_create(e->in_dim, e->out_dim);
}
static void nmmo3_encoder_free_weights(void* weights) { free(weights); }
static void nmmo3_encoder_free_activations(void* activations) { free(activations); }

// ---- Chess constants ----
// Chess uses learnable piece-square and action-context embeddings. The encoder
// reads compact obs bytes, sums the matching rows from board_w/move_context_w
// into hidden space, adds a learned projection of rule metadata, then applies ReLU.

static constexpr int CH_BOARD = 0;
static constexpr int CH_SIDE = 64;
static constexpr int CH_CASTLE = 65;
static constexpr int CH_EP = 69;
static constexpr int CH_RULE50 = 78;
static constexpr int CH_REPETITION = 79;
static constexpr int CH_SELF_CHECK = 80;
static constexpr int CH_OPP_CHECK = 81;
static constexpr int CH_PICK_PHASE = 82;
static constexpr int CH_SELECTED = 83;
static constexpr int CH_VALID_FROM_COUNT = 84;
static constexpr int CH_VALID_FROM = 85;
static constexpr int CH_VALID_TO_COUNT = 101;
static constexpr int CH_VALID_TO = 102;
static constexpr int CH_VALID_PROMOS = 134;
static constexpr int CH_PASS_VALID = 166;
static constexpr int CH_OBS_SIZE = 167;
static constexpr int CH_BOARD_SQUARES = 64;
static constexpr int CH_PIECE_TYPES = 12;
static constexpr int CH_BOARD_FEATURES = CH_BOARD_SQUARES * CH_PIECE_TYPES;
static constexpr int CH_MAX_VALID_FROM = 16;
static constexpr int CH_MAX_VALID_TO = 32;
static constexpr int CH_NULL_SQ = 64;
static constexpr int CH_MAX_MOVE_CONTEXT_ACTIVE = 2 + CH_MAX_VALID_FROM + CH_MAX_VALID_TO;
static constexpr int CH_MOVE_CONTEXT_SELECTED = 0;
static constexpr int CH_MOVE_CONTEXT_VALID_FROM = 64;
static constexpr int CH_MOVE_CONTEXT_VALID_TO = 128;
static constexpr int CH_MOVE_CONTEXT_PHASE = 192;
static constexpr int CH_MOVE_CONTEXT_FEATURES = 194;
static constexpr int CH_META = 51;  // side/castle/ep/clocks/checks/promos/pass

__constant__ int CH_META_SRC[CH_META] = {
    CH_SIDE,
    CH_CASTLE + 0, CH_CASTLE + 1, CH_CASTLE + 2, CH_CASTLE + 3,
    CH_EP + 0, CH_EP + 1, CH_EP + 2, CH_EP + 3, CH_EP + 4,
    CH_EP + 5, CH_EP + 6, CH_EP + 7, CH_EP + 8,
    CH_RULE50,
    CH_REPETITION,
    CH_SELF_CHECK,
    CH_OPP_CHECK,
    CH_VALID_PROMOS + 0, CH_VALID_PROMOS + 1, CH_VALID_PROMOS + 2, CH_VALID_PROMOS + 3,
    CH_VALID_PROMOS + 4, CH_VALID_PROMOS + 5, CH_VALID_PROMOS + 6, CH_VALID_PROMOS + 7,
    CH_VALID_PROMOS + 8, CH_VALID_PROMOS + 9, CH_VALID_PROMOS + 10, CH_VALID_PROMOS + 11,
    CH_VALID_PROMOS + 12, CH_VALID_PROMOS + 13, CH_VALID_PROMOS + 14, CH_VALID_PROMOS + 15,
    CH_VALID_PROMOS + 16, CH_VALID_PROMOS + 17, CH_VALID_PROMOS + 18, CH_VALID_PROMOS + 19,
    CH_VALID_PROMOS + 20, CH_VALID_PROMOS + 21, CH_VALID_PROMOS + 22, CH_VALID_PROMOS + 23,
    CH_VALID_PROMOS + 24, CH_VALID_PROMOS + 25, CH_VALID_PROMOS + 26, CH_VALID_PROMOS + 27,
    CH_VALID_PROMOS + 28, CH_VALID_PROMOS + 29, CH_VALID_PROMOS + 30, CH_VALID_PROMOS + 31,
    CH_PASS_VALID
};

// ---- Chess kernels ----

__global__ void chess_meta_kernel(
        precision_t* __restrict__ meta,
        const precision_t* __restrict__ obs,
        int B, int obs_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * CH_META) return;
    int b = idx / CH_META;
    int m = idx % CH_META;
    int src = CH_META_SRC[m];
    float x = to_float(obs[b * obs_size + src]);
    if (x > 1.0f) x *= (1.0f / 255.0f);
    meta[idx] = from_float(x);
}

__global__ void chess_embed_forward_kernel_opt(
        precision_t* __restrict__ out,
        const precision_t* __restrict__ obs,
        const precision_t* __restrict__ board_w,
        const precision_t* __restrict__ move_context_w,
        const precision_t* __restrict__ bias,
        int B, int hidden, int obs_size) {
    __shared__ int board_features[CH_BOARD_SQUARES]; // 64
    __shared__ int move_context_features[CH_MAX_MOVE_CONTEXT_ACTIVE]; // 50
    __shared__ int move_context_count;

    int b = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    const precision_t* obs_row = obs + b * obs_size;

    int lane = threadIdx.x % 32;
    int board_count = 0;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int sq = i * 32 + lane;
        int piece = (int)to_float(obs_row[CH_BOARD + sq]);
        unsigned mask = __ballot_sync(0xffffffff, piece > 0);
        int lane_offset = __popc(mask & ((1u << lane) - 1));
        if (piece > 0) {
            board_features[board_count + lane_offset] = (piece - 1) * CH_BOARD_SQUARES + sq;
        }
        board_count += __popc(mask); // count bits set to 1
    }

    // retrieve and broadcast across warp
    int phase_val = 0, selected_val = 0, from_count = 0, to_count = 0;
    if (lane == 0) phase_val = (int)to_float(obs_row[CH_PICK_PHASE]) > 0 ? 1 : 0;
    if (lane == 1) selected_val = (int)to_float(obs_row[CH_SELECTED]);
    if (lane == 2) from_count = (int)to_float(obs_row[CH_VALID_FROM_COUNT]);
    if (lane == 3) to_count   = (int)to_float(obs_row[CH_VALID_TO_COUNT]);
    phase_val = __shfl_sync(0xffffffff, phase_val, 0);
    selected_val = __shfl_sync(0xffffffff, selected_val, 1);
    from_count = __shfl_sync(0xffffffff, from_count, 2);
    to_count = __shfl_sync(0xffffffff, to_count, 3);

    int n = 0;
    if (lane == 0) move_context_features[n] = CH_MOVE_CONTEXT_PHASE + phase_val;
    n = 1;
    if (selected_val < CH_BOARD_SQUARES) {
        if (lane == 0) move_context_features[n] = CH_MOVE_CONTEXT_SELECTED + selected_val;
        n++;
    }

    if (lane < from_count) {
        int sq = (int)to_float(obs_row[CH_VALID_FROM + lane]);
        move_context_features[n + lane] = CH_MOVE_CONTEXT_VALID_FROM + sq;
    }
    n += from_count;

    if (lane < to_count) {
        int sq = (int)to_float(obs_row[CH_VALID_TO + lane]);
        move_context_features[n + lane] = CH_MOVE_CONTEXT_VALID_TO + sq;
    }
    n += to_count;
    if (lane == 0) move_context_count = n;

    __syncthreads();
    if (h >= hidden) return;

    int out_idx = b * hidden + h;
    float acc = to_float(out[out_idx]) + to_float(bias[h]);

    for (int i = 0; i < board_count; i++) {
        acc += to_float(board_w[board_features[i] * hidden + h]);
    }

    for (int i = 0; i < move_context_count; i++) {
        acc += to_float(move_context_w[move_context_features[i] * hidden + h]);
    }

    out[out_idx] = from_float(fmaxf(0.0f, acc));

}

__global__ void chess_embed_forward_kernel(
        precision_t* __restrict__ out,
        const precision_t* __restrict__ obs,
        const precision_t* __restrict__ board_w,
        const precision_t* __restrict__ move_context_w,
        const precision_t* __restrict__ bias,
        int B, int hidden, int obs_size) {
    int b = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    const precision_t* obs_row = obs + b * obs_size;

    __shared__ int board_features[CH_BOARD_SQUARES];
    __shared__ int board_count;
    __shared__ int move_context_features[CH_MAX_MOVE_CONTEXT_ACTIVE];
    __shared__ int move_context_count;

    if (threadIdx.x == 0) {
        int n = 0;
        #pragma unroll 8
        for (int sq = 0; sq < CH_BOARD_SQUARES; sq++) {
            int piece = (int)to_float(obs_row[CH_BOARD + sq]);
            if (piece > 0) {
                board_features[n++] = (piece - 1) * CH_BOARD_SQUARES + sq;
            }
        }
        board_count = n;

        n = 0;
        int phase = (int)to_float(obs_row[CH_PICK_PHASE]) > 0 ? 1 : 0;
        move_context_features[n++] = CH_MOVE_CONTEXT_PHASE + phase;

        int selected = (int)to_float(obs_row[CH_SELECTED]);
        if (selected < CH_BOARD_SQUARES) {
            move_context_features[n++] = CH_MOVE_CONTEXT_SELECTED + selected;
        }

        int from_count = (int)to_float(obs_row[CH_VALID_FROM_COUNT]);
        for (int k = 0; k < from_count; k++) {
            int sq = (int)to_float(obs_row[CH_VALID_FROM + k]);
            move_context_features[n++] = CH_MOVE_CONTEXT_VALID_FROM + sq;
        }

        int to_count = (int)to_float(obs_row[CH_VALID_TO_COUNT]);
        for (int k = 0; k < to_count; k++) {
            int sq = (int)to_float(obs_row[CH_VALID_TO + k]);
            move_context_features[n++] = CH_MOVE_CONTEXT_VALID_TO + sq;
        }

        move_context_count = n;
    }

    __syncthreads();
    if (h >= hidden) return;

    int out_idx = b * hidden + h;
    float acc = to_float(out[out_idx]) + to_float(bias[h]);

    for (int i = 0; i < board_count; i++) {
        acc += to_float(board_w[board_features[i] * hidden + h]);
    }

    for (int i = 0; i < move_context_count; i++) {
        acc += to_float(move_context_w[move_context_features[i] * hidden + h]);
    }

    out[out_idx] = from_float(fmaxf(0.0f, acc));
}

__global__ void chess_embed_backward_kernel(
        float* __restrict__ board_wgrad_f,
        float* __restrict__ move_context_wgrad_f,
        const precision_t* __restrict__ grad,
        const precision_t* __restrict__ obs,
        int B, int hidden, int obs_size) {
    int b = blockIdx.y;
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    const precision_t* obs_row = obs + b * obs_size;

    __shared__ int board_features[CH_BOARD_SQUARES];
    __shared__ int board_count;
    __shared__ int move_context_features[CH_MAX_MOVE_CONTEXT_ACTIVE];
    __shared__ int move_context_count;

    if (threadIdx.x == 0) {
        int n = 0;
        #pragma unroll 8
        for (int sq = 0; sq < CH_BOARD_SQUARES; sq++) {
            int piece = (int)to_float(obs_row[CH_BOARD + sq]);
            if (piece > 0) {
                board_features[n++] = (piece - 1) * CH_BOARD_SQUARES + sq;
            }
        }
        board_count = n;

        n = 0;
        int phase = (int)to_float(obs_row[CH_PICK_PHASE]) > 0 ? 1 : 0;
        move_context_features[n++] = CH_MOVE_CONTEXT_PHASE + phase;

        int selected = (int)to_float(obs_row[CH_SELECTED]);
        if (selected < CH_BOARD_SQUARES) {
            move_context_features[n++] = CH_MOVE_CONTEXT_SELECTED + selected;
        }

        int from_count = (int)to_float(obs_row[CH_VALID_FROM_COUNT]);
        for (int k = 0; k < from_count; k++) {
            int sq = (int)to_float(obs_row[CH_VALID_FROM + k]);
            move_context_features[n++] = CH_MOVE_CONTEXT_VALID_FROM + sq;
        }

        int to_count = (int)to_float(obs_row[CH_VALID_TO_COUNT]);
        for (int k = 0; k < to_count; k++) {
            int sq = (int)to_float(obs_row[CH_VALID_TO + k]);
            move_context_features[n++] = CH_MOVE_CONTEXT_VALID_TO + sq;
        }

        move_context_count = n;
    }

    __syncthreads();
    if (h >= hidden) return;

    int grad_idx = b * hidden + h;
    float g = to_float(grad[grad_idx]);
    if (g == 0.0f) return;

    for (int i = 0; i < board_count; i++) {
        atomicAdd(&board_wgrad_f[board_features[i] * hidden + h], g);
    }

    for (int i = 0; i < move_context_count; i++) {
        atomicAdd(&move_context_wgrad_f[move_context_features[i] * hidden + h], g);
    }
}

// ---- Chess encoder structs ----

struct ChessEmbedEncoderWeights {
    PrecisionTensor board_w, move_context_w, meta_w, bias;
    int obs_size, hidden;
};

struct ChessEmbedEncoderActivations {
    PrecisionTensor meta, out, saved_obs;
    PrecisionTensor board_wgrad, move_context_wgrad, meta_wgrad, bgrad;
    FloatTensor board_wgrad_f, move_context_wgrad_f;
};

// ---- Chess encoder interface ----

static ChessEmbedEncoderWeights* chess_embed_encoder_create(int obs_size, int hidden) {
    ChessEmbedEncoderWeights* ew = (ChessEmbedEncoderWeights*)calloc(1, sizeof(ChessEmbedEncoderWeights));
    ew->obs_size = obs_size;
    ew->hidden = hidden;
    return ew;
}

static PrecisionTensor chess_embed_encoder_forward(void* w, void* activations,
        PrecisionTensor input, cudaStream_t stream) {
    ChessEmbedEncoderWeights* ew = (ChessEmbedEncoderWeights*)w;
    ChessEmbedEncoderActivations* a = (ChessEmbedEncoderActivations*)activations;
    int B = input.shape[0];

    if (a->saved_obs.data) puf_copy(&a->saved_obs, &input, stream);
    chess_meta_kernel<<<grid_size(B * CH_META), BLOCK_SIZE, 0, stream>>>(
        a->meta.data, input.data, B, ew->obs_size);
    puf_mm(&a->meta, &ew->meta_w, &a->out, stream);
    dim3 embed_grid(grid_size(ew->hidden), B);
    chess_embed_forward_kernel<<<embed_grid, BLOCK_SIZE, 0, stream>>>(
        a->out.data, input.data, ew->board_w.data, ew->move_context_w.data, ew->bias.data,
        B, ew->hidden, ew->obs_size);
    return a->out;
}

static void chess_embed_encoder_backward(void* w, void* activations,
        PrecisionTensor grad, cudaStream_t stream) {
    ChessEmbedEncoderWeights* ew = (ChessEmbedEncoderWeights*)w;
    ChessEmbedEncoderActivations* a = (ChessEmbedEncoderActivations*)activations;
    int B = grad.shape[0];
    int H = ew->hidden;

    n3_relu_backward_kernel<<<grid_size(B * H), BLOCK_SIZE, 0, stream>>>(
        grad.data, a->out.data, B * H);
    bias_grad_kernel<<<H, 256, 0, stream>>>(a->bgrad.data, grad.data, B, H);
    puf_mm_tn(&grad, &a->meta, &a->meta_wgrad, stream);

    int board_n = CH_BOARD_FEATURES * H;
    int move_context_n = CH_MOVE_CONTEXT_FEATURES * H;
    cudaMemsetAsync(a->board_wgrad_f.data, 0, board_n * sizeof(float), stream);
    cudaMemsetAsync(a->move_context_wgrad_f.data, 0, move_context_n * sizeof(float), stream);
    dim3 embed_grid(grid_size(H), B);
    chess_embed_backward_kernel<<<embed_grid, BLOCK_SIZE, 0, stream>>>(
        a->board_wgrad_f.data, a->move_context_wgrad_f.data, grad.data, a->saved_obs.data,
        B, H, ew->obs_size);
    n3_float_to_precision_kernel<<<grid_size(board_n), BLOCK_SIZE, 0, stream>>>(
        a->board_wgrad.data, a->board_wgrad_f.data, board_n);
    n3_float_to_precision_kernel<<<grid_size(move_context_n), BLOCK_SIZE, 0, stream>>>(
        a->move_context_wgrad.data, a->move_context_wgrad_f.data, move_context_n);
}

static void chess_embed_encoder_init_weights(void* w, uint64_t* seed, cudaStream_t stream) {
    ChessEmbedEncoderWeights* ew = (ChessEmbedEncoderWeights*)w;
    puf_normal_init(&ew->board_w, 0.02f, (*seed)++, stream);
    puf_normal_init(&ew->move_context_w, 0.02f, (*seed)++, stream);
    PrecisionTensor wt = {.data = ew->meta_w.data, .shape = {ew->hidden, CH_META}};
    puf_kaiming_init(&wt, 1.0f, (*seed)++, stream);
    cudaMemsetAsync(ew->bias.data, 0, numel(ew->bias.shape) * sizeof(precision_t), stream);
}

static void chess_embed_encoder_reg_params(void* w, Allocator* alloc) {
    ChessEmbedEncoderWeights* ew = (ChessEmbedEncoderWeights*)w;
    ew->board_w = {.shape = {CH_BOARD_FEATURES, ew->hidden}};
    ew->move_context_w = {.shape = {CH_MOVE_CONTEXT_FEATURES, ew->hidden}};
    ew->meta_w = {.shape = {ew->hidden, CH_META}};
    ew->bias = {.shape = {ew->hidden}};
    alloc_register(alloc, &ew->board_w);
    alloc_register(alloc, &ew->move_context_w);
    alloc_register(alloc, &ew->meta_w);
    alloc_register(alloc, &ew->bias);
}

static void chess_embed_encoder_reg_train(void* w, void* activations,
        Allocator* acts, Allocator* grads, int B_TT) {
    ChessEmbedEncoderWeights* ew = (ChessEmbedEncoderWeights*)w;
    ChessEmbedEncoderActivations* a = (ChessEmbedEncoderActivations*)activations;
    *a = {};
    a->meta = {.shape = {B_TT, CH_META}};
    a->out = {.shape = {B_TT, ew->hidden}};
    a->saved_obs = {.shape = {B_TT, ew->obs_size}};
    alloc_register(acts, &a->meta);
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->saved_obs);

    a->board_wgrad = {.shape = {CH_BOARD_FEATURES, ew->hidden}};
    a->move_context_wgrad = {.shape = {CH_MOVE_CONTEXT_FEATURES, ew->hidden}};
    a->meta_wgrad = {.shape = {ew->hidden, CH_META}};
    a->bgrad = {.shape = {ew->hidden}};
    a->board_wgrad_f = {.shape = {CH_BOARD_FEATURES, ew->hidden}};
    a->move_context_wgrad_f = {.shape = {CH_MOVE_CONTEXT_FEATURES, ew->hidden}};
    alloc_register(grads, &a->board_wgrad);
    alloc_register(grads, &a->move_context_wgrad);
    alloc_register(grads, &a->meta_wgrad);
    alloc_register(grads, &a->bgrad);
    alloc_register(acts, &a->board_wgrad_f);
    alloc_register(acts, &a->move_context_wgrad_f);
}

static void chess_embed_encoder_reg_rollout(void* w, void* activations,
        Allocator* alloc, int B) {
    ChessEmbedEncoderWeights* ew = (ChessEmbedEncoderWeights*)w;
    ChessEmbedEncoderActivations* a = (ChessEmbedEncoderActivations*)activations;
    a->meta = {.shape = {B, CH_META}};
    a->out = {.shape = {B, ew->hidden}};
    alloc_register(alloc, &a->meta);
    alloc_register(alloc, &a->out);
}

static void* chess_embed_encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    return chess_embed_encoder_create(e->in_dim, e->out_dim);
}
static void chess_embed_encoder_free_weights(void* weights) { free(weights); }
static void chess_embed_encoder_free_activations(void* activations) { free(activations); }

// Override encoder vtable for known ocean environments. No-op for unknown envs.
static void create_custom_encoder(const std::string& env_name, Encoder* enc) {
    if (env_name == "nmmo3") {
        *enc = Encoder{
            .forward = nmmo3_encoder_forward,
            .backward = nmmo3_encoder_backward,
            .init_weights = nmmo3_encoder_init_weights,
            .reg_params = nmmo3_encoder_reg_params,
            .reg_train = nmmo3_encoder_reg_train,
            .reg_rollout = nmmo3_encoder_reg_rollout,
            .create_weights = nmmo3_encoder_create_weights,
            .free_weights = nmmo3_encoder_free_weights,
            .free_activations = nmmo3_encoder_free_activations,
            .in_dim = enc->in_dim, .out_dim = enc->out_dim,
            .activation_size = sizeof(NMMO3EncoderActivations),
        };
    } else if (env_name == "chess") {
        *enc = Encoder{
            .forward = chess_embed_encoder_forward,
            .backward = chess_embed_encoder_backward,
            .init_weights = chess_embed_encoder_init_weights,
            .reg_params = chess_embed_encoder_reg_params,
            .reg_train = chess_embed_encoder_reg_train,
            .reg_rollout = chess_embed_encoder_reg_rollout,
            .create_weights = chess_embed_encoder_create_weights,
            .free_weights = chess_embed_encoder_free_weights,
            .free_activations = chess_embed_encoder_free_activations,
            .in_dim = enc->in_dim, .out_dim = enc->out_dim,
            .activation_size = sizeof(ChessEmbedEncoderActivations),
        };
    }
}
