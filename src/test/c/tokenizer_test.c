#include "tokenizer.h"
#include <string.h>
#include <assert.h>
#include "s8.h"
#include "str.h"

int test_tokenizer_encode_a(void)
{
    tokenizer_t tokenizer = {0};
    int ret = tokenizer_create(&tokenizer, "src/test/resources/vocab.txt");
    assert(ret == 0);

    int offset = 0;
    const trie_t *node = trie_longest(&tokenizer.trie, (uint8_t *)"a", 1, &offset);
    assert(node->value == 1037);
    assert(offset == 1);

    da_u32 out_ids = {0};
    ret = tokenizer_encode(tokenizer, (uint8_t *)"a", 1, &out_ids);
    assert(ret == 0);

    u32 expected_ids[] = {101, 1037, 102};
    da_u32 expected_ids_da = {0};
    da_u32_extend(&expected_ids_da, expected_ids, 3);
    assert(da_u32_eq(&out_ids, &expected_ids_da));

    da_u32_free(&out_ids);
    da_u32_free(&expected_ids_da);
    tokenizer_destroy(&tokenizer);
    return 0;
}

int test_tokenizer_encode_2(void)
{
    tokenizer_t tokenizer = {0};
    int ret = tokenizer_create(&tokenizer, "src/test/resources/vocab.txt");
    assert(ret == 0);

    da_u32 out_ids = {0};
    const char *text = "hello world";
    ret = tokenizer_encode(tokenizer, (uint8_t *)text, strlen(text), &out_ids);
    assert(ret == 0);
    // 101, 7592, 2088, 2050, 102,
    u32 expected_ids[] = {101, 7592, 2088, 102};
    da_u32 expected_ids_da = {0};
    da_u32_extend(&expected_ids_da, expected_ids, 4);
    assert(da_u32_eq(&out_ids, &expected_ids_da));
    da_u32_free(&expected_ids_da);

    s8 text2 = s8_init("hello worlda");

    da_u32_reset(&out_ids);
    ret = tokenizer_encode(tokenizer, (uint8_t *)text2.data, text2.len, &out_ids);
    assert(ret == 0);
    // 101, 7592, 2088, 2050, 102,
    u32 expected_ids2[] = {101, 7592, 2088, 2050, 102};
    da_u32 expected_ids_da2 = {0};
    da_u32_extend(&expected_ids_da2, expected_ids2, 5);
    assert(da_u32_eq(&out_ids, &expected_ids_da2));

    text2 = s8_init("what is my name?");
    da_u32_reset(&out_ids);
    ret = tokenizer_encode(tokenizer, (uint8_t *)text2.data, text2.len, &out_ids);
    assert(ret == 0);
    // 101, 2054, 2003, 2026, 2171, 1029, 102,
    u32 expected_ids3[] = {101, 2054, 2003, 2026, 2171, 1029, 102};
    da_u32 expected_ids_da3 = {0};
    da_u32_extend(&expected_ids_da3, expected_ids3, 7);
    assert(da_u32_eq(&out_ids, &expected_ids_da3));

    da_u32_free(&out_ids);
    da_u32_free(&expected_ids_da);
    da_u32_free(&expected_ids_da2);
    da_u32_free(&expected_ids_da3);
    tokenizer_destroy(&tokenizer);
    return 0;
}

int test_str_split(void)
{
    s8 input = s8_init("hello world test");
    s8 delimiter = s8_init(" ");
    da_s8 result = str_split(input, delimiter);
    assert(result.len == 3);
    assert(s8_eq(result.data[0], s8_init("hello")));
    assert(s8_eq(result.data[1], s8_init("world")));
    assert(s8_eq(result.data[2], s8_init("test")));
    da_s8_free(&result);
    return 0;
}

int main(void)
{
    test_tokenizer_encode_a();
    test_tokenizer_encode_2();
    test_str_split();
    return 0;
}