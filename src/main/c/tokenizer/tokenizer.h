#pragma once
#include <stdbool.h>
#include "trie.h"
#include "da.h"
DA(u32)

typedef struct tokenizer_t
{
    trie_t trie;
} tokenizer_t;

int tokenizer_create(tokenizer_t *tok, const char *vocab_txt_path);
int tokenizer_encode(const tokenizer_t tok, uint8_t *text, int text_len, da_u32 *out_ids);
void tokenizer_destroy(tokenizer_t *tok);