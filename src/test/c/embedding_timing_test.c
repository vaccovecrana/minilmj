#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "minilm.h"
#include "tbf.h"

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main(int argc, char **argv) {
    const char *tbf_path = "src/test/resources/bert_weights.tbf";
    const char *vocab_path = "src/test/resources/vocab.txt";
    const char *test_text = "what's the capital of germany?";
    
    if (argc > 1) {
        test_text = argv[1];
    }
    
    printf("Loading model...\n");
    double load_start = get_time_ms();
    minilm_t m;
    if (minilm_create(&m, tbf_path, vocab_path) != 0) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    double load_time = get_time_ms() - load_start;
    printf("Model loading time: %.2f ms\n", load_time);
    
    printf("\nEmbedding text: '%s'\n", test_text);
    
    // Warm-up run
    tensor_t warmup;
    minilm_embed(m, (char *)test_text, strlen(test_text), &warmup);
    tensor_destroy(&warmup);
    
    // Actual timed run
    double embed_start = get_time_ms();
    tensor_t out;
    t_status status = minilm_embed(m, (char *)test_text, strlen(test_text), &out);
    double embed_time = get_time_ms() - embed_start;
    
    if (status != T_OK) {
        fprintf(stderr, "Embedding failed with status: %d\n", status);
        minilm_destroy(&m);
        return 1;
    }
    
    printf("Embedding time: %.2f ms\n", embed_time);
    printf("Embedding size: %u\n", out.dims[0]);
    
    // Verify embedding is valid (not all zeros, no NaN)
    float sum = 0.0f;
    int nan_count = 0;
    for (size_t i = 0; i < out.dims[0]; i++) {
        if (out.data[i] != out.data[i]) { // NaN check
            nan_count++;
        }
        sum += out.data[i];
    }
    printf("Embedding sum: %.6f\n", sum);
    printf("NaN count: %d\n", nan_count);
    
    tensor_destroy(&out);
    minilm_destroy(&m);
    
    printf("\n✓ Test completed\n");
    return 0;
}

