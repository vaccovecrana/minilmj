#pragma once
#include <stddef.h>
#include "tensor.h"

/// ```python
/// out = weights[ids]
/// ```
void nn_embeddings_forward(tensor_t *out, const uint32_t *ids, size_t num_tokens, tensor_t weights);

/// ```python
/// out = (x - mean(x)) / std(x) * gamma + beta
/// ```
t_status nn_layer_norm_forward(tensor_t *out, tensor_t x_tensor, tensor_t gamma, tensor_t beta);

/// ```python
/// out = x @ weights + bias
/// ```
t_status nn_linear_forward(tensor_t *out,    // [S,HIDDEN_SIZE]
                           tensor_t x,       // [S,HIDDEN_SIZE]
                           tensor_t weights, // [HIDDEN_SIZE, HIDDEN_SIZE]
                           tensor_t bias);   // [1, HIDDEN_SIZE]

/// PyTorch reference:
/// ```python
/// def dot_product_attention(query, key, value):
///     scale_factor = 1 / math.sqrt(query.size(-1))
///     attn_weight = query @ key.transpose(-2, -1) * scale_factor
///     attn_weight = torch.softmax(attn_weight, dim=-1)
///     return attn_weight @ value
/// ```
t_status nn_dot_product_attention_forward(
    tensor_t *out,
    tensor_t query_tensor,
    tensor_t key_tensor,
    tensor_t value_tensor,
    uint32_t n_attention_heads);

/// mean with mask
void nn_mean_pooling(tensor_t *out, tensor_t in, tensor_t attention_mask);

/// ```python
/// out = (t - mean(t)) / std(t)
/// ```
void nn_normalize(tensor_t *t);