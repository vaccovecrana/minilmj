#include "minilm.h"
#include "tbf.h"
#include "s8.h"
#include "tokenizer.h"
#include "nn.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

// Forward declarations for internal functions
extern tensor_t minilm_embedder_forward(da_u32 ids, minilm_t weights);
extern t_status tensor_bmm(tensor_t *out, tensor_t A, tensor_t B);
extern void nn_softmax(tensor_t t);

// Test to verify attention mask is being applied correctly
int main() {
    minilm_t m;
    if (minilm_create(&m, "src/test/resources/bert_weights.tbf", "src/test/resources/vocab.txt") != 0) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    
    // Tokenize a short sequence
    const char *test = "berlin";
    s8 str = s8_from_parts((char *)test, strlen(test));
    da_u32 ids = {0};
    if (minilm_tokenize(m, str, &ids) != T_OK) {
        fprintf(stderr, "Failed to tokenize\n");
        return 1;
    }
    
    printf("Tokenized '%s': %zu tokens\n", test, ids.len);
    printf("First 10 token IDs: ");
    for (size_t i = 0; i < ids.len && i < 10; i++) {
        printf("%u ", ids.data[i]);
    }
    printf("\n");
    
    // Count padding
    size_t padding = 0;
    size_t non_padding = 0;
    for (size_t i = 0; i < ids.len; i++) {
        if (ids.data[i] == 0) padding++;
        else non_padding++;
    }
    printf("Non-padding tokens: %zu, Padding tokens: %zu (out of %zu total)\n", non_padding, padding, ids.len);
    
    // Get embeddings
    tensor_t embedder_out = minilm_embedder_forward(ids, m);
    printf("\nEmbedder output shape: [%u, %u]\n", embedder_out.dims[0], embedder_out.dims[1]);
    
    // Test first encoder layer to check attention mask
    tensor_t q, k, v;
    nn_linear_forward(&q, embedder_out, m.attention[0].query, m.attention[0].query_bias);
    nn_linear_forward(&k, embedder_out, m.attention[0].key, m.attention[0].key_bias);
    nn_linear_forward(&v, embedder_out, m.attention[0].value, m.attention[0].value_bias);
    
    printf("\nQ shape: [%u, %u], K shape: [%u, %u], V shape: [%u, %u]\n",
           q.dims[0], q.dims[1], k.dims[0], k.dims[1], v.dims[0], v.dims[1]);
    
    // Manually compute attention scores to check mask
    uint32_t num_tokens_actual = q.dims[0];
    uint32_t num_heads = 12;
    uint32_t head_size = q.dims[1] / num_heads;
    
    printf("\nnum_tokens_actual: %u, num_heads: %u, head_size: %u\n", 
           num_tokens_actual, num_heads, head_size);
    
    // Reshape for attention
    uint32_t dims[3] = {num_tokens_actual, num_heads, head_size};
    tensor_t qt = tensor_view(3, dims, q.data);
    tensor_t kt = tensor_view(3, dims, k.data);
    
    tensor_t qtv_T, kt_T;
    tensor_t qt_final, kt_final;
    tensor_permute(&qtv_T, qt, 0, 1); // [12, num_tokens, 32]
    tensor_permute(&qt_final, qtv_T, 1, 2); // [12, 32, num_tokens]
    tensor_permute(&kt_T, kt, 0, 1);  // [12, num_tokens, 32]
    tensor_permute(&kt_final, kt_T, 1, 2);  // [12, 32, num_tokens]
    
    tensor_t out_tensor_3d;
    tensor_bmm(&out_tensor_3d, qtv_T, kt_final);
    
    float scale = 1.0f / sqrtf((float)head_size);
    tensor_unary_op(out_tensor_3d, U_SCALE, &scale);
    
    printf("\nAttention scores shape: [%u, %u, %u]\n", 
           out_tensor_3d.dims[0], out_tensor_3d.dims[1], out_tensor_3d.dims[2]);
    printf("Strides: [%lu, %lu, %lu]\n", 
           out_tensor_3d.strides[0], out_tensor_3d.strides[1], out_tensor_3d.strides[2]);
    
    // Check some attention scores before masking
    printf("\nBefore masking - Sample attention scores (head 0, query 0):\n");
    for (size_t key_idx = 0; key_idx < 10 && key_idx < out_tensor_3d.dims[2]; key_idx++) {
        size_t idx = 0 * out_tensor_3d.strides[0] + 0 * out_tensor_3d.strides[1] + key_idx * out_tensor_3d.strides[2];
        printf("  key[%zu] (token_id=%u): %.6f\n", key_idx, ids.data[key_idx], out_tensor_3d.data[idx]);
    }
    
    // Apply mask
    const float MASK_VALUE = -1e9f;
    size_t num_tokens = ids.len;
    size_t masked_count = 0;
    
    for (size_t head = 0; head < out_tensor_3d.dims[0]; head++) {
        for (size_t query_idx = 0; query_idx < out_tensor_3d.dims[1]; query_idx++) {
            for (size_t key_idx = 0; key_idx < out_tensor_3d.dims[2]; key_idx++) {
                size_t idx = head * out_tensor_3d.strides[0] + 
                            query_idx * out_tensor_3d.strides[1] + 
                            key_idx * out_tensor_3d.strides[2];
                
                bool query_is_padding = (query_idx < num_tokens && ids.data[query_idx] == 0);
                bool key_is_padding = (key_idx < num_tokens && ids.data[key_idx] == 0);
                
                if (query_is_padding || key_is_padding) {
                    float old_val = out_tensor_3d.data[idx];
                    out_tensor_3d.data[idx] = MASK_VALUE;
                    if (old_val != MASK_VALUE) masked_count++;
                }
            }
        }
    }
    
    printf("\nMasked %zu positions\n", masked_count);
    printf("Expected to mask: ~%zu positions (most of %u * %u = %u)\n", 
           (size_t)(out_tensor_3d.dims[0] * out_tensor_3d.dims[1] * out_tensor_3d.dims[2] * 0.98),
           out_tensor_3d.dims[0], out_tensor_3d.dims[1] * out_tensor_3d.dims[2],
           out_tensor_3d.dims[0] * out_tensor_3d.dims[1] * out_tensor_3d.dims[2]);
    
    // Check after masking
    printf("\nAfter masking - Sample attention scores (head 0, query 0):\n");
    for (size_t key_idx = 0; key_idx < 10 && key_idx < out_tensor_3d.dims[2]; key_idx++) {
        size_t idx = 0 * out_tensor_3d.strides[0] + 0 * out_tensor_3d.strides[1] + key_idx * out_tensor_3d.strides[2];
        bool is_masked = (out_tensor_3d.data[idx] == MASK_VALUE);
        printf("  key[%zu] (token_id=%u): %.6f %s\n", 
               key_idx, ids.data[key_idx], out_tensor_3d.data[idx],
               is_masked ? "[MASKED]" : "");
    }
    
    // Apply softmax to one row and check
    printf("\nApplying softmax to head 0, query 0:\n");
    tensor_t a = tensor_slice(&out_tensor_3d, 0, 0, true);  // [1, num_tokens, num_tokens] for head 0
    printf("Tensor a shape after slicing head: [%u, %u, %u]\n", a.dims[0], a.dims[1], a.dims[2]);
    
    // Try slicing dim 1 (queries) - this should give us [1, 256] -> but we want [256]
    // Actually, we need to slice dim 0 first to remove the singleton dimension
    tensor_t a_no_head = tensor_slice(&a, 0, 0, false);  // [256, 256] - remove head dimension
    printf("Tensor a_no_head shape: [%u, %u]\n", a_no_head.dims[0], a_no_head.dims[1]);
    tensor_t b = tensor_slice(&a_no_head, 0, 0, false);  // [256] - slice query dimension
    printf("Tensor b shape after slicing query: [%u] (should be 256)\n", b.dims[0]);
    printf("Before softmax: ");
    for (size_t i = 0; i < 10 && i < b.dims[0]; i++) {
        printf("%.6f ", b.data[i]);
    }
    printf("\n");
    
    nn_softmax(b);
    
    printf("After softmax: ");
    float sum = 0.0f;
    float sum_all = 0.0f;
    for (size_t i = 0; i < 10 && i < b.dims[0]; i++) {
        printf("%.6f ", b.data[i]);
        sum += b.data[i];
    }
    printf("\n");
    printf("Sum of first 10: %.6f (should be <= 1.0)\n", sum);
    
    // Sum all values
    for (size_t i = 0; i < b.dims[0]; i++) {
        sum_all += b.data[i];
    }
    printf("Sum of ALL %u values: %.6f (should be 1.0)\n", b.dims[0], sum_all);
    
    // Check if padding positions are near zero after softmax
    printf("\nSoftmax values for padding vs non-padding (head 0, query 0):\n");
    for (size_t key_idx = 0; key_idx < 10 && key_idx < b.dims[0]; key_idx++) {
        printf("  key[%zu] (token_id=%u, is_padding=%d): %.10f\n",
               key_idx, ids.data[key_idx], (ids.data[key_idx] == 0), b.data[key_idx]);
    }
    
    // Check a few padding positions further out
    printf("\nSoftmax values for padding positions (head 0, query 0, indices 100-109):\n");
    for (size_t key_idx = 100; key_idx < 110 && key_idx < b.dims[0]; key_idx++) {
        printf("  key[%zu] (token_id=%u, is_padding=%d): %.10f\n",
               key_idx, ids.data[key_idx], (ids.data[key_idx] == 0), b.data[key_idx]);
    }
    
    // Count how many are effectively zero (very small)
    size_t near_zero_count = 0;
    for (size_t i = 0; i < b.dims[0]; i++) {
        if (b.data[i] < 1e-10f) {
            near_zero_count++;
        }
    }
    printf("\nPositions with value < 1e-10 (effectively zero): %zu out of %u\n", 
           near_zero_count, b.dims[0]);
    
    da_u32_free(&ids);
    tensor_destroy(&embedder_out);
    tensor_destroy(&q);
    tensor_destroy(&k);
    tensor_destroy(&v);
    tensor_destroy(&out_tensor_3d);
    minilm_destroy(&m);
    
    return 0;
}

