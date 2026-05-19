// Correctness + speed test for chess_embed_forward_kernel vs chess_embed_forward_kernel_opt.
// Build (from repo root):
// nvcc -O2 -arch=native -std=c++17 -I. -Isrc   -DOBS_TENSOR_T=uint8_t -DENV_NAME=chess   -Xcompiler=-DPLATFORM_DESKTOP -Xcompiler=-fopenmp   src/chess_test.cu -lcublas -lcurand -lcudnn -lm -lpthread -o chess_test

#define PRECISION_FLOAT

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <string>

#include "models.cu"
#include "ocean.cu"

static const int WARMUP_ITERS = 200;
static const int TIMING_ITERS = 2000;

static inline float randf() { return (float)rand() / RAND_MAX; }

static void fill_random_obs(float* host_obs, int B, int obs_size) {
    for (int b = 0; b < B; b++) {
        float* row = host_obs + b * obs_size;
        for (int i = 0; i < obs_size; i++) row[i] = 0.0f;

        // Board squares [0..63]: piece index 0 (empty) or 1..12
        // Place ~20 random pieces per board
        int npieces = 16 + rand() % 16;
        for (int p = 0; p < npieces; p++) {
            int sq = rand() % CH_BOARD_SQUARES;
            row[CH_BOARD + sq] = (float)(1 + rand() % CH_PIECE_TYPES);
        }

        row[CH_SIDE] = (float)(rand() % 2);

        for (int i = 0; i < 4; i++)
            row[CH_CASTLE + i] = (float)(rand() % 2);

        int ep_file = rand() % 9;
        row[CH_EP + ep_file] = 1.0f;

        row[CH_RULE50] = (float)(rand() % 256);
        row[CH_REPETITION] = (rand() % 3 == 0) ? 255.0f : (rand() % 3 == 0) ? 128.0f : 0.0f;
        row[CH_SELF_CHECK] = (rand() % 5 == 0) ? 255.0f : 0.0f;
        row[CH_OPP_CHECK] = (rand() % 5 == 0) ? 255.0f : 0.0f;

        row[CH_PICK_PHASE] = (float)(rand() % 2);

        int selected = (rand() % 3 == 0) ? (rand() % CH_BOARD_SQUARES) : CH_NULL_SQ;
        row[CH_SELECTED] = (float)selected;

        int from_count = 1 + rand() % CH_MAX_VALID_FROM;
        row[CH_VALID_FROM_COUNT] = (float)from_count;
        for (int k = 0; k < from_count; k++)
            row[CH_VALID_FROM + k] = (float)(rand() % CH_BOARD_SQUARES);

        int to_count = 1 + rand() % CH_MAX_VALID_TO;
        row[CH_VALID_TO_COUNT] = (float)to_count;
        for (int k = 0; k < to_count; k++)
            row[CH_VALID_TO + k] = (float)(rand() % CH_BOARD_SQUARES);

        for (int i = 0; i < 32; i++)
            row[CH_VALID_PROMOS + i] = (float)(rand() % 2);

        row[CH_PASS_VALID] = (rand() % 2) ? 255.0f : 0.0f;
    }
}

struct ChessEmbedTest {
    precision_t *d_obs, *d_board_w, *d_move_context_w, *d_bias;
    precision_t *d_out_ref, *d_out_opt;
    float *h_obs;
    int B, hidden, obs_size;
};

static ChessEmbedTest* create_test(int B, int hidden) {
    auto* t = (ChessEmbedTest*)calloc(1, sizeof(ChessEmbedTest));
    t->B = B;
    t->hidden = hidden;
    t->obs_size = CH_OBS_SIZE;

    int obs_total = B * CH_OBS_SIZE;
    int board_total = CH_BOARD_FEATURES * hidden;
    int mc_total = CH_MOVE_CONTEXT_FEATURES * hidden;
    int out_total = B * hidden;

    t->h_obs = (float*)malloc(obs_total * sizeof(float));
    fill_random_obs(t->h_obs, B, CH_OBS_SIZE);

    cudaMalloc(&t->d_obs, obs_total * sizeof(precision_t));
    cudaMalloc(&t->d_board_w, board_total * sizeof(precision_t));
    cudaMalloc(&t->d_move_context_w, mc_total * sizeof(precision_t));
    cudaMalloc(&t->d_bias, hidden * sizeof(precision_t));
    cudaMalloc(&t->d_out_ref, out_total * sizeof(precision_t));
    cudaMalloc(&t->d_out_opt, out_total * sizeof(precision_t));

    // Upload obs
    precision_t* tmp = (precision_t*)malloc(obs_total * sizeof(precision_t));
    for (int i = 0; i < obs_total; i++) tmp[i] = (precision_t)t->h_obs[i];
    cudaMemcpy(t->d_obs, tmp, obs_total * sizeof(precision_t), cudaMemcpyHostToDevice);
    free(tmp);

    // Random weights
    auto fill_rand = [](precision_t* dst, int n) {
        float* buf = (float*)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) buf[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        precision_t* tmp = (precision_t*)malloc(n * sizeof(precision_t));
        for (int i = 0; i < n; i++) tmp[i] = (precision_t)buf[i];
        cudaMemcpy(dst, tmp, n * sizeof(precision_t), cudaMemcpyHostToDevice);
        free(buf);
        free(tmp);
    };
    fill_rand(t->d_board_w, board_total);
    fill_rand(t->d_move_context_w, mc_total);
    fill_rand(t->d_bias, hidden);

    return t;
}

static void destroy_test(ChessEmbedTest* t) {
    cudaFree(t->d_obs);
    cudaFree(t->d_board_w);
    cudaFree(t->d_move_context_w);
    cudaFree(t->d_bias);
    cudaFree(t->d_out_ref);
    cudaFree(t->d_out_opt);
    free(t->h_obs);
    free(t);
}

// Prepare out buffer with a known "meta_w matmul" stand-in so both kernels
// start from identical state. Fills with small random values.
static void reset_out(precision_t* d_out, int count, unsigned seed) {
    srand(seed);
    float* buf = (float*)malloc(count * sizeof(float));
    for (int i = 0; i < count; i++) buf[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    precision_t* tmp = (precision_t*)malloc(count * sizeof(precision_t));
    for (int i = 0; i < count; i++) tmp[i] = (precision_t)buf[i];
    cudaMemcpy(d_out, tmp, count * sizeof(precision_t), cudaMemcpyHostToDevice);
    free(buf);
    free(tmp);
}

static bool test_correctness(int B, int hidden) {
    printf("=== Correctness test (B=%d, hidden=%d) ===\n", B, hidden);
    auto* t = create_test(B, hidden);
    int out_total = B * hidden;

    unsigned seed = 12345;
    reset_out(t->d_out_ref, out_total, seed);
    reset_out(t->d_out_opt, out_total, seed);

    dim3 grid(grid_size(hidden), B);

    chess_embed_forward_kernel<<<grid, BLOCK_SIZE>>>(
        t->d_out_ref, t->d_obs, t->d_board_w, t->d_move_context_w, t->d_bias,
        B, hidden, CH_OBS_SIZE);

    chess_embed_forward_kernel_opt<<<grid, BLOCK_SIZE>>>(
        t->d_out_opt, t->d_obs, t->d_board_w, t->d_move_context_w, t->d_bias,
        B, hidden, CH_OBS_SIZE);

    cudaDeviceSynchronize();

    float* h_ref = (float*)malloc(out_total * sizeof(float));
    float* h_opt = (float*)malloc(out_total * sizeof(float));
    cudaMemcpy(h_ref, t->d_out_ref, out_total * sizeof(precision_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_opt, t->d_out_opt, out_total * sizeof(precision_t), cudaMemcpyDeviceToHost);

    float max_abs_err = 0, max_rel_err = 0;
    int mismatches = 0;
    for (int i = 0; i < out_total; i++) {
        float ref = h_ref[i], opt = h_opt[i];
        float abs_err = fabsf(ref - opt);
        float rel_err = (fabsf(ref) > 1e-6f) ? abs_err / fabsf(ref) : abs_err;
        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
        if (abs_err > 1e-4f) {
            if (mismatches < 10) {
                int b = i / hidden, h = i % hidden;
                printf("  MISMATCH [b=%d, h=%d]: ref=%.6f opt=%.6f diff=%.6f\n",
                       b, h, ref, opt, abs_err);
            }
            mismatches++;
        }
    }

    bool pass = (mismatches == 0);
    printf("  max_abs_err = %.6e, max_rel_err = %.6e\n", max_abs_err, max_rel_err);
    if (pass)
        printf("  PASSED (%d elements match)\n\n", out_total);
    else
        printf("  FAILED (%d / %d mismatches)\n\n", mismatches, out_total);

    free(h_ref);
    free(h_opt);
    destroy_test(t);
    return pass;
}

struct BenchArgs {
    ChessEmbedTest* t;
    bool use_opt;
};

static void run_kernel(BenchArgs* args) {
    auto* t = args->t;
    dim3 grid(grid_size(t->hidden), t->B);
    precision_t* out = args->use_opt ? t->d_out_opt : t->d_out_ref;
    if (args->use_opt) {
        chess_embed_forward_kernel_opt<<<grid, BLOCK_SIZE>>>(
            out, t->d_obs, t->d_board_w, t->d_move_context_w, t->d_bias,
            t->B, t->hidden, CH_OBS_SIZE);
    } else {
        chess_embed_forward_kernel<<<grid, BLOCK_SIZE>>>(
            out, t->d_obs, t->d_board_w, t->d_move_context_w, t->d_bias,
            t->B, t->hidden, CH_OBS_SIZE);
    }
}

static float bench_kernel(ChessEmbedTest* t, bool use_opt) {
    BenchArgs args = {t, use_opt};
    int out_total = t->B * t->hidden;
    unsigned seed = 99999;

    for (int i = 0; i < WARMUP_ITERS; i++) {
        reset_out(use_opt ? t->d_out_opt : t->d_out_ref, out_total, seed);
        run_kernel(&args);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < TIMING_ITERS; i++) {
        run_kernel(&args);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / TIMING_ITERS;
}

static void test_speed(int B, int hidden) {
    printf("=== Speed test (B=%d, hidden=%d) ===\n", B, hidden);
    auto* t = create_test(B, hidden);

    float ref_ms = bench_kernel(t, false);
    float opt_ms = bench_kernel(t, true);

    printf("  reference:  %8.2f us\n", ref_ms * 1000);
    printf("  optimized:  %8.2f us\n", opt_ms * 1000);
    if (opt_ms > 0 && ref_ms > 0)
        printf("  speedup:    %.2fx\n", ref_ms / opt_ms);
    printf("\n");

    destroy_test(t);
}

int main(int argc, char** argv) {
    int B = 1024;
    int hidden = 256;
    bool run_correctness = true;
    bool run_speed = true;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) B = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) hidden = atoi(argv[++i]);
        else if (strcmp(argv[i], "correctness") == 0) { run_correctness = true; run_speed = false; }
        else if (strcmp(argv[i], "speed") == 0) { run_correctness = false; run_speed = true; }
        else if (strcmp(argv[i], "all") == 0) { run_correctness = true; run_speed = true; }
        else {
            printf("Usage: %s [correctness|speed|all] [--batch N] [--hidden N]\n", argv[0]);
            return 1;
        }
    }

    printf("chess_embed_forward_kernel test\n");
    printf("  B=%d, hidden=%d, obs_size=%d\n", B, hidden, CH_OBS_SIZE);
    printf("  board_features=%d, move_context_features=%d\n\n",
           CH_BOARD_FEATURES, CH_MOVE_CONTEXT_FEATURES);

    // Warm up GPU
    float* dummy;
    cudaMalloc(&dummy, 64 * 1024 * 1024);
    for (int i = 0; i < 50; i++) cudaMemset(dummy, 0, 64 * 1024 * 1024);
    cudaDeviceSynchronize();
    cudaFree(dummy);

    srand(42);

    bool pass = true;
    if (run_correctness) {
        pass &= test_correctness(B, hidden);
        pass &= test_correctness(1, hidden);
        pass &= test_correctness(B, 128);
    }

    if (run_speed) {
        test_speed(B, hidden);
        test_speed(4096, hidden);
        test_speed(B, 512);
    }

    return pass ? 0 : 1;
}
