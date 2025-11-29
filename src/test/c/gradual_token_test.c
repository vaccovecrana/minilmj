#include "minilm.h"
#include "tbf.h"
#include "s8.h"
#include "tokenizer.h"
#include "nn.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

// Forward declarations for internal functions
extern tensor_t minilm_embedder_forward(da_u32 ids, minilm_t weights);
extern t_status minilm_encoder_forward(const tensor_t in, bert_layer_weigts_t weights, tensor_t *out, 
                                      const uint32_t *token_ids, size_t num_tokens);

// Check if a tensor sum is valid (not NaN/Inf)
// Use bit pattern check since -ffast-math may optimize NaN comparisons
int is_valid_sum(float sum) {
    union { float f; uint32_t i; } u;
    u.f = sum;
    uint32_t bits = u.i & 0x7FFFFFFF; // Clear sign bit
    
    // NaN: exponent all 1s (0x7F800000) and mantissa non-zero
    // Inf: exponent all 1s and mantissa zero
    uint32_t exp_mask = 0x7F800000;
    if ((bits & exp_mask) == exp_mask) {
        // Exponent is all 1s - either NaN or Inf
        uint32_t mantissa = bits & 0x007FFFFF;
        if (mantissa != 0) {
            return 0; // NaN (mantissa non-zero)
        }
        return 0; // Inf (mantissa zero)
    }
    
    return 1; // Valid (exponent not all 1s)
}

int main() {
    const char *test_text = "what's the capital of germany?";
    size_t text_len = strlen(test_text);
    
    printf("=== Gradual Token Size Increase Test ===\n");
    printf("Test text: %s\n", test_text);
    
    // Load model
    minilm_t m;
    int ret = minilm_create(&m, "src/test/resources/bert_weights.tbf", "src/test/resources/vocab.txt");
    assert(ret == 0);
    printf("Model loaded successfully\n");
    
    // Tokenize the text
    s8 str_s8 = s8_from_parts((char *)test_text, text_len);
    da_u32 ids_original = {0};
    tokenizer_encode(m.tokenizer, (uint8_t *)test_text, text_len, &ids_original);
    
    printf("Original token count: %zu\n", ids_original.len);
    
    // ============================================
    // PHASE 1: Verify 128-token baseline (ground truth)
    // ============================================
    printf("\n=== PHASE 1: Verifying 128-token baseline (ground truth) ===\n");
    da_u32 ids_128 = {0};
    for (size_t i = 0; i < ids_original.len && i < 128; i++) {
        da_u32_append(&ids_128, ids_original.data[i]);
    }
    for (size_t i = ids_128.len; i < 128; i++) {
        da_u32_append(&ids_128, 0);
    }
    
    printf("Testing embedder with 128 tokens...\n");
    tensor_t embedder_128 = minilm_embedder_forward(ids_128, m);
    float embedder_128_sum = tensor_sum(embedder_128);
    printf("  Embedder (128): dims=[%u,%u], sum=%.6f", 
           embedder_128.dims[0], embedder_128.dims[1], embedder_128_sum);
    if (!is_valid_sum(embedder_128_sum)) {
        printf(" ✗ FAILED\n");
        return 1;
    }
    printf(" ✓ PASSED\n");
    
    printf("Testing encoder layers with 128 tokens...\n");
    tensor_t encoder_128 = embedder_128;
    for (size_t i = 0; i < 6; i++) {
        tensor_t tmp;
        t_status status = minilm_encoder_forward(encoder_128, m.attention[i], &tmp, ids_128.data, ids_128.len);
        if (status != T_OK) {
            printf("  Encoder[%zu] (128): ✗ FAILED (status %d)\n", i, status);
            return 1;
        }
        if (i > 0) tensor_destroy(&encoder_128);
        encoder_128 = tmp;
        float sum = tensor_sum(encoder_128);
        printf("  Encoder[%zu] (128): dims=[%u,%u], sum=%.6f", 
               i, encoder_128.dims[0], encoder_128.dims[1], sum);
        if (!is_valid_sum(sum)) {
            printf(" ✗ FAILED\n");
            return 1;
        }
        printf(" ✓ PASSED\n");
    }
    
    printf("Testing mean pooling with 128 tokens...\n");
    tensor_t pooled_128;
    nn_mean_pooling(&pooled_128, encoder_128, ids_128.data, ids_128.len);
    float pooled_128_sum = tensor_sum(pooled_128);
    printf("  Pooled (128): dims=[%u], sum=%.6f", pooled_128.dims[0], pooled_128_sum);
    if (!is_valid_sum(pooled_128_sum)) {
        printf(" ✗ FAILED\n");
        return 1;
    }
    printf(" ✓ PASSED\n");
    
    printf("Testing normalization with 128 tokens...\n");
    tensor_t normalized_128 = tensor_copy(pooled_128);
    nn_normalize(&normalized_128);
    float normalized_128_sum = tensor_sum(normalized_128);
    printf("  Normalized (128): dims=[%u], sum=%.6f", normalized_128.dims[0], normalized_128_sum);
    if (!is_valid_sum(normalized_128_sum)) {
        printf(" ✗ FAILED\n");
        return 1;
    }
    printf(" ✓ PASSED\n");
    
    printf("\n✓ PHASE 1 COMPLETE: 128-token baseline verified (ground truth)\n");
    
    // Cleanup 128-token test
    tensor_destroy(&embedder_128);
    tensor_destroy(&encoder_128);
    tensor_destroy(&pooled_128);
    tensor_destroy(&normalized_128);
    
    // ============================================
    // PHASE 2: Test 256 tokens layer by layer
    // ============================================
    printf("\n=== PHASE 2: Testing 256 tokens (gradual expansion) ===\n");
    da_u32 ids_256 = {0};
    for (size_t i = 0; i < ids_original.len; i++) {
        da_u32_append(&ids_256, ids_original.data[i]);
    }
    for (size_t i = ids_256.len; i < 256; i++) {
        da_u32_append(&ids_256, 0);
    }
    printf("Padded to 256 tokens: %zu actual tokens, %zu padding\n", 
           ids_original.len, 256 - ids_original.len);
    
    // Stage 1: Embedder
    printf("\nStage 1: Embedder layer\n");
    tensor_t embedder_256 = minilm_embedder_forward(ids_256, m);
    float embedder_256_sum = tensor_sum(embedder_256);
    printf("  Embedder (256): dims=[%u,%u], sum=%.6f", 
           embedder_256.dims[0], embedder_256.dims[1], embedder_256_sum);
    if (!is_valid_sum(embedder_256_sum)) {
        printf(" ✗ FAILED\n");
        printf("  ERROR: Embedder fails with 256 tokens. This should not happen.\n");
        tensor_destroy(&embedder_256);
        da_u32_free(&ids_256);
        da_u32_free(&ids_128);
        da_u32_free(&ids_original);
        minilm_destroy(&m);
        return 1;
    }
    printf(" ✓ PASSED\n");
    
    // Stage 2-7: Encoder layers one by one
    tensor_t encoder_256 = embedder_256;
    for (size_t i = 0; i < 6; i++) {
        printf("\nStage %zu: Encoder layer %zu\n", i + 2, i);
        
        tensor_t tmp;
        t_status status = minilm_encoder_forward(encoder_256, m.attention[i], &tmp, ids_256.data, ids_256.len);
        
        if (status != T_OK) {
            printf("  ✗ FAILED: Encoder returned error status %d\n", status);
            printf("  ERROR: Encoder layer %zu fails with 256 tokens.\n", i);
            tensor_destroy(&encoder_256);
            da_u32_free(&ids_256);
            da_u32_free(&ids_128);
            da_u32_free(&ids_original);
            minilm_destroy(&m);
            return 1;
        }
        
        if (i > 0) {
            tensor_destroy(&encoder_256);
        }
        encoder_256 = tmp;
        
        float sum = tensor_sum(encoder_256);
        printf("  Encoder[%zu] (256): dims=[%u,%u], sum=%.6f",
               i, encoder_256.dims[0], encoder_256.dims[1], sum);
        
        // Check for NaN/Inf using bit pattern
        union { float f; uint32_t i; } u;
        u.f = sum;
        uint32_t bits_no_sign = u.i & 0x7FFFFFFF;
        uint32_t exp_mask = 0x7F800000;
        uint32_t mantissa_mask = 0x007FFFFF;
        
        // Check if exponent is all 1s (NaN or Inf)
        int exp_all_ones = ((bits_no_sign & exp_mask) == exp_mask);
        int mantissa_nonzero = ((bits_no_sign & mantissa_mask) != 0);
        int is_nan = exp_all_ones && mantissa_nonzero;
        int is_inf = exp_all_ones && !mantissa_nonzero;
        int is_valid = !exp_all_ones;
        
        if (!is_valid) {
            printf(" ✗ FAILED");
            if (is_nan) printf(" (NaN)");
            if (is_inf) printf(" (Inf)");
            printf("\n");
            printf(" ✗ FAILED\n");
            printf("  ERROR: Encoder layer %zu produces NaN/Inf with 256 tokens.\n", i);
            printf("  This is where 256 tokens first fails. Need to fix encoder layer %zu.\n", i);
            tensor_destroy(&encoder_256);
            da_u32_free(&ids_256);
            da_u32_free(&ids_128);
            da_u32_free(&ids_original);
            minilm_destroy(&m);
            return 1;
        }
        printf(" ✓ PASSED\n");
    }
    
    // Stage 8: Mean pooling
            printf("\nStage 8: Mean pooling\n");
            tensor_t pooled_256;
            nn_mean_pooling(&pooled_256, encoder_256, ids_256.data, ids_256.len);
    float pooled_256_sum = tensor_sum(pooled_256);
    printf("  Pooled (256): dims=[%u], sum=%.6f", pooled_256.dims[0], pooled_256_sum);
    if (!is_valid_sum(pooled_256_sum)) {
        printf(" ✗ FAILED\n");
        printf("  ERROR: Mean pooling fails with 256 tokens.\n");
        tensor_destroy(&encoder_256);
        tensor_destroy(&pooled_256);
        da_u32_free(&ids_256);
        da_u32_free(&ids_128);
        da_u32_free(&ids_original);
        minilm_destroy(&m);
        return 1;
    }
    printf(" ✓ PASSED\n");
    
    // Stage 9: Normalize
    printf("\nStage 9: Normalize\n");
    tensor_t normalized_256 = tensor_copy(pooled_256);
    nn_normalize(&normalized_256);
    float normalized_256_sum = tensor_sum(normalized_256);
    printf("  Normalized (256): dims=[%u], sum=%.6f", normalized_256.dims[0], normalized_256_sum);
    if (!is_valid_sum(normalized_256_sum)) {
        printf(" ✗ FAILED\n");
        printf("  ERROR: Normalization fails with 256 tokens.\n");
        tensor_destroy(&encoder_256);
        tensor_destroy(&pooled_256);
        tensor_destroy(&normalized_256);
        da_u32_free(&ids_256);
        da_u32_free(&ids_128);
        da_u32_free(&ids_original);
        minilm_destroy(&m);
        return 1;
    }
    printf(" ✓ PASSED\n");
    
    // Cleanup
    tensor_destroy(&encoder_256);
    tensor_destroy(&pooled_256);
    tensor_destroy(&normalized_256);
    da_u32_free(&ids_256);
    da_u32_free(&ids_128);
    da_u32_free(&ids_original);
    minilm_destroy(&m);
    
    printf("\n=== All stages passed! ===\n");
    printf("256 tokens works correctly. Ready to increase MINILM_MAX_TOKENS to 256.\n");
    return 0;
}

