#include "nn.h"
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "tensor.h"

void nn_embeddings_forward(tensor_t *out, const uint32_t *ids, size_t num_tokens, tensor_t weights)
{
    *out = tensor_create(2, (uint32_t[]){num_tokens, weights.dims[1]});
    for (size_t i = 0; i < num_tokens; i++)
    {
        uint32_t id = ids[i];
        tensor_t dst_view = tensor_slice(out, 0, i, true);
        tensor_t src_view = tensor_slice(&weights, 0, id, true);
        memcpy(dst_view.data, src_view.data, src_view.dims[1] * sizeof(float));
    }
}

float nn_mean(tensor_t x)
{
    return tensor_sum(x) / (float)x.dims[0];
}

t_status nn_layer_norm_forward(
    tensor_t *out,
    tensor_t x,
    tensor_t gamma, // [1, HIDDEN_SIZE]
    tensor_t beta)
{
    float eps = 1e-12f;
    // assert(gamma.cols == HIDDEN_SIZE && beta.cols == HIDDEN_SIZE);

    float *xd = x.data;

    for (size_t t = 0; t < x.dims[0]; ++t)
    {
        tensor_t x_view = tensor_slice(&x, 0, t, false);

        // // 1) mean over features
        float mean = nn_mean(x_view);

        // // 2) variance over features
        tensor_t x_view_copy = tensor_copy(x_view);
        tensor_unary_op(x_view_copy, U_SUB, &mean);
        float op_param = 2.0f;
        tensor_unary_op(x_view_copy, U_POW, &op_param);
        float var = tensor_sum(x_view_copy);
        var /= (float)x_view_copy.dims[0];

        // 3) normalize, then scale & shift
        float inv_std = 1.0f / sqrtf(var + eps);
        tensor_t x_view_copy2 = tensor_copy(x_view);
        tensor_unary_op(x_view_copy2, U_SUB, &mean);
        tensor_unary_op(x_view_copy2, U_SCALE, &inv_std);
        tensor_binary_op(x_view_copy2, gamma, B_MUL);
        tensor_binary_op(x_view_copy2, beta, B_ADD);

        tensor_t out_view = tensor_slice(out, 0, t, true);
        memcpy(out_view.data, x_view_copy2.data, tensor_numel(x_view_copy2) * sizeof(float));

        tensor_destroy(&x_view_copy);
        tensor_destroy(&x_view_copy2);
    }
    return T_OK;
}

t_status nn_linear_forward(tensor_t *out,    // [S,HIDDEN_SIZE]
                           tensor_t x,       // [S,HIDDEN_SIZE]
                           tensor_t weights, // [HIDDEN_SIZE, HIDDEN_SIZE]
                           tensor_t bias     // [1, HIDDEN_SIZE]
)
{
    tensor_t weights_T;
    m_try(tensor_permute(&weights_T, weights, 0, 1));
    m_try(tensor_matmul(out, x, weights_T));
    tensor_destroy(&weights_T);
    tensor_binary_op(*out, bias, B_ADD);
    return T_OK;
}
void nn_softmax(tensor_t t)
{
    // Numerical stability: subtract max before exp to prevent overflow
    // This doesn't change the result (softmax is shift-invariant) but prevents exp() overflow
    float max_val = t.data[0];
    for (size_t i = 1; i < t.dims[0]; i++) {
        if (t.data[i] > max_val) {
            max_val = t.data[i];
        }
    }
    
    // Subtract max from all values
    for (size_t i = 0; i < t.dims[0]; i++) {
        t.data[i] -= max_val;
    }
    
    // Now apply exp - values are in range [-inf, 0], so exp() won't overflow
    tensor_unary_op(t, U_EXP, NULL);
    
    // Normalize
    float sum = tensor_sum(t);
    
    // Check for NaN/Inf in sum - use bit pattern check
    union { float f; uint32_t i; } u_sum;
    u_sum.f = sum;
    uint32_t sum_bits_no_sign = u_sum.i & 0x7FFFFFFF;
    uint32_t exp_mask = 0x7F800000;
    int sum_is_nan_or_inf = ((sum_bits_no_sign & exp_mask) == exp_mask);
    
    if (sum_is_nan_or_inf) {
        // Sum is NaN or Inf - set to uniform distribution
        float uniform = 1.0f / (float)t.dims[0];
        for (size_t i = 0; i < t.dims[0]; i++) {
            t.data[i] = uniform;
        }
        return;
    }
    
    // Check for zero or very small sum
    if (sum <= 0.0f || sum < 1e-10f || !(sum == sum)) {  // !(sum == sum) checks for NaN using comparison
        // Sum too small or zero (all values were very negative, exp() underflowed to 0) - set to uniform distribution
        float uniform = 1.0f / (float)t.dims[0];
        for (size_t i = 0; i < t.dims[0]; i++) {
            t.data[i] = uniform;
        }
        return;
    }
    
    float scale = 1.0f / sum;
    
    // Check if scale is inf or NaN
    union { float f; uint32_t i; } u_scale;
    u_scale.f = scale;
    uint32_t scale_bits_no_sign = u_scale.i & 0x7FFFFFFF;
    if ((scale_bits_no_sign & exp_mask) == exp_mask) {
        // Scale is NaN or Inf - set to uniform distribution
        float uniform = 1.0f / (float)t.dims[0];
        for (size_t i = 0; i < t.dims[0]; i++) {
            t.data[i] = uniform;
        }
        return;
    }
    
    tensor_unary_op(t, U_SCALE, &scale);
}

#define m_len(x) (sizeof(x) / sizeof(x[0]))

t_status tensor_bmm(tensor_t *out, tensor_t A, tensor_t B)
{
    *out = tensor_create(3, (uint32_t[]){A.dims[0], A.dims[1], B.dims[2]});
    for (size_t i = 0; i < A.dims[0]; i++)
    {
        tensor_t a = tensor_slice(&A, 0, i, false);
        tensor_t b = tensor_slice(&B, 0, i, false);
        tensor_t tmp;
        m_try(tensor_matmul(&tmp, a, b));

        tensor_t out_view = tensor_slice(out, 0, i, false);
        memcpy(out_view.data, tmp.data, tensor_numel(tmp) * sizeof(float));
        tensor_destroy(&tmp);
    }
    return T_OK;
}

t_status nn_dot_product_attention_forward(
    tensor_t *out,
    tensor_t query_tensor,
    tensor_t key_tensor,
    tensor_t value_tensor,
    uint32_t n_attention_heads,
    const uint32_t *token_ids,
    size_t num_tokens)
{
    int ret;
    uint32_t num_tokens_actual = query_tensor.dims[0];
    uint32_t num_heads = n_attention_heads;
    uint32_t head_size = query_tensor.dims[1] / num_heads;
    
    uint32_t dims[3] = {num_tokens_actual, num_heads, head_size};
    tensor_t qt = tensor_view(m_len(dims), dims, query_tensor.data);
    tensor_t kt = tensor_view(m_len(dims), dims, key_tensor.data);
    tensor_t vt = tensor_view(m_len(dims), dims, value_tensor.data);

    tensor_t qtv_T, kt_T, vt_T;
    tensor_t qt_final, kt_final, vt_final;
    m_try(tensor_permute(&qtv_T, qt, 0, 1)); // [12, num_tokens, 32]
    m_try(tensor_permute(&qt_final, qtv_T, 1, 2)); // [12, 32, num_tokens]
    m_try(tensor_permute(&kt_T, kt, 0, 1));  // [12, num_tokens, 32]
    m_try(tensor_permute(&kt_final, kt_T, 1, 2));  // [12, 32, num_tokens]
    m_try(tensor_permute(&vt_T, vt, 0, 1));  // [12, num_tokens, 32]
    m_try(tensor_permute(&vt_final, vt_T, 1, 2));  // [12, 32, num_tokens]

    tensor_t out_tensor_3d;

    m_try(tensor_bmm(&out_tensor_3d, qtv_T, kt_final));

    float scale = 1.0f / sqrtf((float)head_size);
    tensor_unary_op(out_tensor_3d, U_SCALE, &scale);

    // Apply attention mask before softmax
    // For padding tokens (token_ids[k] == 0), set attention scores to large negative value
    // so they become 0 after softmax
    const float MASK_VALUE = -1e9f;
    // out_tensor_3d has shape [num_heads, num_tokens_actual, num_tokens_actual] = [12, num_tokens_actual, num_tokens_actual]
    // Tensor is row-major, so index = head * stride[0] + query * stride[1] + key * stride[2]
    // For contiguous tensor: stride[0] = dims[1] * dims[2], stride[1] = dims[2], stride[2] = 1
    // num_tokens_actual should equal num_tokens (both are the padded length)
    for (size_t head = 0; head < out_tensor_3d.dims[0]; head++)  // For each head
    {
        for (size_t query_idx = 0; query_idx < out_tensor_3d.dims[1]; query_idx++)  // For each query token
        {
            for (size_t key_idx = 0; key_idx < out_tensor_3d.dims[2]; key_idx++)  // For each key token
            {
                // Calculate index using strides (row-major layout)
                size_t idx = head * out_tensor_3d.strides[0] + 
                            query_idx * out_tensor_3d.strides[1] + 
                            key_idx * out_tensor_3d.strides[2];
                
                // Mask if query token is padding OR key token is padding
                // Both query_idx and key_idx should be < num_tokens (the padded length)
                bool query_is_padding = (query_idx < num_tokens && token_ids[query_idx] == 0);
                bool key_is_padding = (key_idx < num_tokens && token_ids[key_idx] == 0);
                
                if (query_is_padding || key_is_padding) {
                    out_tensor_3d.data[idx] = MASK_VALUE;
                }
            }
        }
    }

    // softmax
    // out_tensor_3d shape: [num_heads, num_tokens, num_tokens] = [12, 256, 256]
    // dim 0 = heads, dim 1 = queries, dim 2 = keys
    for (size_t i = 0; i < out_tensor_3d.dims[0]; i++)
    {
        // Get head i: [num_tokens, num_tokens] (drop head dimension)
        tensor_t a = tensor_slice(&out_tensor_3d, 0, i, false);  // [num_tokens, num_tokens] for head i
        for (size_t j = 0; j < out_tensor_3d.dims[1]; j++)
        {
            // Slice dimension 0 (query tokens) at index j to get all keys for this query
            // This gives us [num_tokens] - attention scores for head i, query j across all keys
            tensor_t b = tensor_slice(&a, 0, j, false);  // [num_tokens] - attention scores for head i, query j
            nn_softmax(b);
        }
    }

    tensor_t attn_tensor;

    m_try(tensor_bmm(&attn_tensor, out_tensor_3d, vt_T));

    tensor_t attn_tensor_T;
    m_try(tensor_permute(&attn_tensor_T, attn_tensor, 0, 1));

    tensor_destroy(&attn_tensor);
    tensor_destroy(&out_tensor_3d);
    tensor_destroy(&qtv_T);
    tensor_destroy(&kt_T);
    tensor_destroy(&vt_T);
    tensor_destroy(&qt_final);
    tensor_destroy(&kt_final);
    tensor_destroy(&vt_final);

    // Create a copy instead of a view so we can destroy attn_tensor_T
    tensor_t attn_tensor_view = tensor_view(2, (uint32_t[]){num_tokens, 384}, attn_tensor_T.data);
    *out = tensor_copy(attn_tensor_view);
    tensor_destroy(&attn_tensor_T);
    return T_OK;
}

void nn_mean_pooling(tensor_t *out, tensor_t in, const uint32_t *token_ids, size_t num_tokens)
{
    // Count non-padding tokens (tokens with id != 0)
    size_t non_padding_count = 0;
    for (size_t i = 0; i < num_tokens; i++) {
        if (token_ids[i] != 0) {
            non_padding_count++;
        }
    }
    
    if (non_padding_count == 0) {
        // All tokens are padding - return zero tensor
        *out = tensor_create(1, (uint32_t[]){in.dims[1]});
        for (size_t i = 0; i < in.dims[1]; i++) {
            out->data[i] = 0.0f;
        }
        return;
    }
    
    // Find first non-padding token
    size_t first_non_padding_idx = 0;
    while (first_non_padding_idx < num_tokens && token_ids[first_non_padding_idx] == 0) {
        first_non_padding_idx++;
    }
    
    // Start with first non-padding row
    tensor_t first_row = tensor_slice(&in, 0, first_non_padding_idx, false);  // [HIDDEN_SIZE]
    *out = tensor_copy(first_row);
    
    // Sum all other non-padding rows
    for (size_t i = first_non_padding_idx + 1; i < num_tokens; i++) {
        if (token_ids[i] != 0) {
            tensor_t row = tensor_slice(&in, 0, i, false);  // [HIDDEN_SIZE]
            tensor_binary_op(*out, row, B_ADD);
        }
    }
    
    // Average by dividing by non-padding count
    float scale = 1.0f / (float)non_padding_count;
    tensor_unary_op(*out, U_SCALE, &scale);
}

void nn_normalize(tensor_t *t)
{
    float p = 2.0f;

    tensor_t t_pow = tensor_copy(*t);
    tensor_unary_op(t_pow, U_POW, &p);

    float norm = tensor_sum(t_pow);
    norm = powf(norm, 1.0f / p);
    float scale = 1.0f / norm;
    tensor_unary_op(*t, U_SCALE, &scale);

    tensor_destroy(&t_pow);
}