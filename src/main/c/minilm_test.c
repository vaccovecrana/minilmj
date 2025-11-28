#include "minilm.h"
#include "tbf.h"
#include "nn.h"
#include "s8.h"
#include "tokenizer.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
DA(tensor_t)

void test_query()
{

    minilm_t m;
    minilm_create(&m, "../assets/bert_weights.tbf", "../assets/vocab.txt");

    da_s8 str_list = {0};
    da_s8_append(&str_list, m_s8("paris"));
    da_s8_append(&str_list, m_s8("london"));
    da_s8_append(&str_list, m_s8("berlin"));
    da_s8_append(&str_list, m_s8("madrid"));
    da_s8_append(&str_list, m_s8("rome"));

    da_tensor_t out_list = {0};
    for (size_t i = 0; i < str_list.len; i++)
    {
        tensor_t out;
        minilm_embed(m, (char *)str_list.data[i].data, str_list.data[i].len, &out);
        da_tensor_t_append(&out_list, out);
    }

    const char *query_str = "what's the capital of germany?";
    tensor_t query;
    minilm_embed(m, (char *)query_str, strlen(query_str), &query);

    minilm_destroy(&m);

    int smallest_index = -1;
    float smallest_diff = 1e10;
    for (size_t i = 0; i < out_list.len; i++)
    {
        tensor_t out = out_list.data[i];
        tensor_binary_op(out, query, B_SUB);
        float pow = 2.0f;
        tensor_unary_op(out, U_POW, &pow);
        float diff = tensor_sum(out);
        if (diff < smallest_diff)
        {
            smallest_diff = diff;
            smallest_index = i;
        }
    }
    printf("query: %s\n", query_str);
    printf("answer: %s\n", str_list.data[smallest_index].data);
}

void test_a()
{
    minilm_t m;
    minilm_create(&m, "../assets/bert_weights.tbf", "../assets/vocab.txt");

    tensor_t out;
    minilm_embed(m, "a", 1, &out);
    
    printf("Embedding 'a' - shape: [");
    for (int i = 0; i < out.ndim; i++) {
        printf("%u", out.dims[i]);
        if (i < out.ndim - 1) printf(", ");
    }
    printf("], first 5 values: ");
    for (int i = 0; i < 5 && i < out.dims[0]; i++) {
        printf("%.6f ", out.data[i]);
    }
    printf("\n");

    minilm_destroy(&m);

    FILE *fp = fopen("../src/str_a.bin", "rb");
    if (!fp)
    {
        fprintf(stderr, "Failed to open file\n");
        exit(1);
    }
    tensor_t test_tensor;
    tensor_load(fp, &test_tensor);
    fclose(fp);

    tensor_binary_op(test_tensor, out, B_SUB);
    printf("diff\n");
    tensor_unary_op(test_tensor, U_ABS, NULL);
    float diff = tensor_sum(test_tensor);
    printf("diff: %f\n", diff);
    assert(diff < 1e-4);

    tensor_destroy(&out);
    tensor_destroy(&test_tensor);
}

int main(int argc, char **argv)
{
    test_query();
    test_a();
    return 0;
}