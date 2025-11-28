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
        // Debug: check dimensions before slice
        if (t >= x.dims[0]) {
            fprintf(stderr, "[C] layer_norm: Invalid slice t=%zu, x.dims[0]=%u\n", t, x.dims[0]);
            return T_ERR;
        }
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
    tensor_unary_op(t, U_EXP, NULL);
    float sum = tensor_sum(t);
    float scale = 1.0f / sum;
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
    uint32_t n_attention_heads)
{
    int ret;
    uint32_t num_tokens = query_tensor.dims[0];
    uint32_t num_heads = n_attention_heads;
    uint32_t head_size = query_tensor.dims[1] / num_heads;
    uint32_t dims[3] = {num_tokens, num_heads, head_size};
    tensor_t qt = tensor_view(m_len(dims), dims, query_tensor.data);
    tensor_t kt = tensor_view(m_len(dims), dims, key_tensor.data);
    tensor_t vt = tensor_view(m_len(dims), dims, value_tensor.data);

    tensor_t qtv_T, kt_T, vt_T;
    m_try(tensor_permute(&qtv_T, qt, 0, 1)); // [12, 128, 32]
    m_try(tensor_permute(&qt, qtv_T, 1, 2)); // [12, 32, 128]
    m_try(tensor_permute(&kt_T, kt, 0, 1));  // [12, 128, 32]
    m_try(tensor_permute(&kt, kt_T, 1, 2));  // [12, 32, 128]
    m_try(tensor_permute(&vt_T, vt, 0, 1));  // [12, 128, 32]
    m_try(tensor_permute(&vt, vt_T, 1, 2));  // [12, 32, 128]

    tensor_t out_tensor_3d;

    m_try(tensor_bmm(&out_tensor_3d, qtv_T, kt));

    float scale = 1.0f / sqrtf((float)head_size);
    tensor_unary_op(out_tensor_3d, U_SCALE, &scale);

    // softmax

    for (size_t i = 0; i < out_tensor_3d.dims[0]; i++)
    {
        tensor_t a = tensor_slice(&out_tensor_3d, 0, i, true);
        for (size_t j = 0; j < out_tensor_3d.dims[1]; j++)
        {
            tensor_t b = tensor_slice(&a, 1, j, true);
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

    tensor_t attn_tensor_view = tensor_view(2, (uint32_t[]){num_tokens, 384}, attn_tensor_T.data);
    *out = attn_tensor_view;
    return T_OK;
}

void nn_mean_pooling(tensor_t *out, tensor_t in, tensor_t attention_mask)
{
    tensor_t row_0 = tensor_slice(&in, 0, 0, true);
    tensor_t row_1 = tensor_slice(&in, 0, 1, true);
    tensor_t row_2 = tensor_slice(&in, 0, 2, true);

    *out = tensor_copy(row_0);

    // average
    tensor_binary_op(*out, row_1, B_ADD);
    tensor_binary_op(*out, row_2, B_ADD);
    float scale = 1.0f / 3.0f;
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