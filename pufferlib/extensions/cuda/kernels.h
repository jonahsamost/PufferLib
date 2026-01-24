// kernels.h - Non-templated wrapper declarations for CUDA kernel launch functions
// This header allows .cpp files to call kernel launchers

#ifndef PUFFERLIB_KERNELS_H
#define PUFFERLIB_KERNELS_H

#include <cuda_runtime.h>
#include <cstdint>
#include <c10/util/BFloat16.h>

// Float wrappers
void launch_mingru_gate_inference_float(float* out, float* next_state, const float* combined, const float* state_in, int H, int B, cudaStream_t stream);
void launch_log_coeffs_and_values_float(float* log_coeffs, float* log_values, const float* gate, const float* hidden, int N, cudaStream_t stream);
void launch_log_coeffs_and_values_backward_float(float* grad_gate, float* grad_hidden, const float* grad_log_coeffs, const float* grad_log_values, const float* gate, const float* hidden, int N, cudaStream_t stream);
void launch_rmsnorm_forward_float(float* out, float* inv_norm_buf, const float* x, const float* weight, double eps, int T_total, int H, int B, cudaStream_t stream);
void launch_rmsnorm_backward_float(float* grad_x, float* grad_weight, const float* grad_out, const float* inv_norm_buf, const float* x_buf, const float* weight, double eps, int T_total, int H, int B, cudaStream_t stream);
void launch_fused_scan_forward_float(float* out, float* next_state, float* a_star, float* s_vals, float* log_values_buf, const float* combined, const float* state, int T_seq, int H, int B, cudaStream_t stream);
void launch_fused_scan_backward_float(float* grad_combined, float* grad_state, const float* grad_out, const float* grad_next_state, const float* combined, const float* state, const float* a_star_buf, const float* s_buf, const float* log_values_buf, int T_seq, int H, int B, cudaStream_t stream);
void launch_logcumsumexp_forward_float(float* out, double* s_buf, const float* x, int T_total, int H, int B, cudaStream_t stream);
void launch_logcumsumexp_backward_float(float* grad_x, const float* grad_out, const float* x, const double* s_buf, int T_total, int H, int B, cudaStream_t stream);
void launch_ppo_loss_forward_float(float* loss_output, double* saved_for_backward, const float* logits, const float* values_pred, const int64_t* actions, const float* old_logprobs, const float* advantages, const float* prio, const float* values, const float* returns, const float* adv_mean, const float* adv_std, double clip_coef, double vf_clip_coef, double vf_coef, double ent_coef, int T_seq, int A, int N, cudaStream_t stream);
void launch_ppo_loss_backward_float(float* grad_logits, float* grad_values_pred, const float* grad_loss, const float* logits, const int64_t* actions, const float* old_logprobs, const float* advantages, const float* prio, const float* values, const float* returns, const double* saved_for_backward, const float* adv_mean, const float* adv_std, double clip_coef, double vf_clip_coef, double vf_coef, double ent_coef, int T_seq, int A, int N, cudaStream_t stream);
void launch_ppo_loss_forward_optimized_float(float* loss_output, double* saved_for_backward, const float* logits, const float* values_pred, const int64_t* actions, const float* old_logprobs, const float* advantages, const float* prio, const float* values, const float* returns, const float* adv_mean, const float* adv_std, float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef, int T_seq, int A, int N, cudaStream_t stream);
void launch_ppo_loss_backward_optimized_float(float* grad_logits, float* grad_values_pred, const float* grad_loss, const float* logits, const float* values_pred, const int64_t* actions, const float* old_logprobs, const float* advantages, const float* prio, const float* values, const float* returns, const float* adv_mean, const float* adv_std, float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef, int T_seq, int A, int N, cudaStream_t stream);
void launch_sample_logits_float(double* actions, float* logprobs, float* value_out, const float* logits, const float* value, uint64_t seed, const int64_t* offset_ptr, int A, int B, int logits_stride, int value_stride, cudaStream_t stream);

// BFloat16 wrappers
void launch_mingru_gate_inference_bf16(at::BFloat16* out, at::BFloat16* next_state, const at::BFloat16* combined, const at::BFloat16* state_in, int H, int B, cudaStream_t stream);
void launch_log_coeffs_and_values_bf16(at::BFloat16* log_coeffs, at::BFloat16* log_values, const at::BFloat16* gate, const at::BFloat16* hidden, int N, cudaStream_t stream);
void launch_log_coeffs_and_values_backward_bf16(at::BFloat16* grad_gate, at::BFloat16* grad_hidden, const at::BFloat16* grad_log_coeffs, const at::BFloat16* grad_log_values, const at::BFloat16* gate, const at::BFloat16* hidden, int N, cudaStream_t stream);
void launch_rmsnorm_forward_bf16(at::BFloat16* out, float* inv_norm_buf, const at::BFloat16* x, const at::BFloat16* weight, double eps, int T_total, int H, int B, cudaStream_t stream);
void launch_rmsnorm_backward_bf16(at::BFloat16* grad_x, at::BFloat16* grad_weight, const at::BFloat16* grad_out, const float* inv_norm_buf, const at::BFloat16* x_buf, const at::BFloat16* weight, double eps, int T_total, int H, int B, cudaStream_t stream);
void launch_fused_scan_forward_bf16(at::BFloat16* out, at::BFloat16* next_state, float* a_star, float* s_vals, float* log_values_buf, const at::BFloat16* combined, const at::BFloat16* state, int T_seq, int H, int B, cudaStream_t stream);
void launch_fused_scan_backward_bf16(at::BFloat16* grad_combined, at::BFloat16* grad_state, const at::BFloat16* grad_out, const at::BFloat16* grad_next_state, const at::BFloat16* combined, const at::BFloat16* state, const float* a_star_buf, const float* s_buf, const float* log_values_buf, int T_seq, int H, int B, cudaStream_t stream);
void launch_logcumsumexp_forward_bf16(at::BFloat16* out, double* s_buf, const at::BFloat16* x, int T_total, int H, int B, cudaStream_t stream);
void launch_logcumsumexp_backward_bf16(at::BFloat16* grad_x, const at::BFloat16* grad_out, const at::BFloat16* x, const double* s_buf, int T_total, int H, int B, cudaStream_t stream);
void launch_ppo_loss_forward_bf16(float* loss_output, double* saved_for_backward, const at::BFloat16* logits, const at::BFloat16* values_pred, const int64_t* actions, const at::BFloat16* old_logprobs, const at::BFloat16* advantages, const at::BFloat16* prio, const at::BFloat16* values, const at::BFloat16* returns, const float* adv_mean, const float* adv_std, double clip_coef, double vf_clip_coef, double vf_coef, double ent_coef, int T_seq, int A, int N, cudaStream_t stream);
void launch_ppo_loss_backward_bf16(at::BFloat16* grad_logits, at::BFloat16* grad_values_pred, const float* grad_loss, const at::BFloat16* logits, const int64_t* actions, const at::BFloat16* old_logprobs, const at::BFloat16* advantages, const at::BFloat16* prio, const at::BFloat16* values, const at::BFloat16* returns, const double* saved_for_backward, const float* adv_mean, const float* adv_std, double clip_coef, double vf_clip_coef, double vf_coef, double ent_coef, int T_seq, int A, int N, cudaStream_t stream);
void launch_ppo_loss_forward_optimized_bf16(float* loss_output, double* saved_for_backward, const at::BFloat16* logits, const at::BFloat16* values_pred, const int64_t* actions, const at::BFloat16* old_logprobs, const at::BFloat16* advantages, const at::BFloat16* prio, const at::BFloat16* values, const at::BFloat16* returns, const float* adv_mean, const float* adv_std, float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef, int T_seq, int A, int N, cudaStream_t stream);
void launch_ppo_loss_backward_optimized_bf16(at::BFloat16* grad_logits, at::BFloat16* grad_values_pred, const float* grad_loss, const at::BFloat16* logits, const at::BFloat16* values_pred, const int64_t* actions, const at::BFloat16* old_logprobs, const at::BFloat16* advantages, const at::BFloat16* prio, const at::BFloat16* values, const at::BFloat16* returns, const float* adv_mean, const float* adv_std, float clip_coef, float vf_clip_coef, float vf_coef, float ent_coef, int T_seq, int A, int N, cudaStream_t stream);
void launch_sample_logits_bf16(double* actions, at::BFloat16* logprobs, at::BFloat16* value_out, const at::BFloat16* logits, const at::BFloat16* value, uint64_t seed, const int64_t* offset_ptr, int A, int B, int logits_stride, int value_stride, cudaStream_t stream);

#endif // PUFFERLIB_KERNELS_H
