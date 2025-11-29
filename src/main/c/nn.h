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
/// def dot_product_attention(query, key, value, attention_mask=None):
///     scale_factor = 1 / math.sqrt(query.size(-1))
///     attn_weight = query @ key.transpose(-2, -1) * scale_factor
///     if attention_mask is not None:
///         attn_weight = attn_weight.masked_fill(attention_mask == 0, float('-inf'))
///     attn_weight = torch.softmax(attn_weight, dim=-1)
///     return attn_weight @ value
/// ```
/// @param token_ids token IDs array [num_tokens] - used to create attention mask (padding tokens have id == 0)
/// @param num_tokens number of tokens
t_status nn_dot_product_attention_forward(
    tensor_t *out,
    tensor_t query_tensor,
    tensor_t key_tensor,
    tensor_t value_tensor,
    uint32_t n_attention_heads,
    const uint32_t *token_ids,
    size_t num_tokens);

/// mean pooling - averages all non-padding tokens (tokens with id != 0)
/// @param out output tensor [HIDDEN_SIZE]
/// @param in input tensor [num_tokens, HIDDEN_SIZE]
/// @param token_ids token IDs array [num_tokens] - used to identify padding tokens (id == 0)
/// @param num_tokens number of tokens
void nn_mean_pooling(tensor_t *out, tensor_t in, const uint32_t *token_ids, size_t num_tokens);

/// ```python
/// out = (t - mean(t)) / std(t)
/// ```
void nn_normalize(tensor_t *t);