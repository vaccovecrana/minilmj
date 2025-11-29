#include "tokenizer.h"
#include <stdio.h>
#include <assert.h>

#include "trie.h"
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include "s8.h"
#include "str.h"

void str_trim(char *s, const char *trim_chars)
{
    if (trim_chars == NULL)
        trim_chars = " \t\n";
    char *end = s + strlen(s) - 1;
    while (end >= s && strchr(trim_chars, *end))
        end--;
    end[1] = '\0';
}

int tokenizer_create(tokenizer_t *tok, const char *vocab_txt_path)
{
    // Initialize the trie structure to a clean state
    tok->trie = (trie_t){0};

    FILE *fp = fopen(vocab_txt_path, "r");
    if (!fp)
    {
        fprintf(stderr, "Failed to open vocab file\n");
        return 1;
    }

    char line[100];
    int i = 0;
    while (true)
    {
        char *n = fgets(line, sizeof(line), fp);
        if (n == NULL)
            break;
        str_trim(line, " \r\n\t");
        if (line[0] != '[')
        {
            int ret = trie_insert(&tok->trie, (uint8_t *)line, i);
            if (ret != 0)
            {
                fprintf(stderr, "Failed to insert token\n");
                return 1;
            }
        }
        i++;
    }
    fclose(fp);
    return 0;
}

int tokenizer_encode(const tokenizer_t tok, uint8_t *text, int text_len, da_u32 *out_ids)
{
    const trie_t *continuation_tree = trie_find_child(&tok.trie, '#');
    continuation_tree = trie_find_child(continuation_tree, '#');

    if (continuation_tree == NULL)
    {
        fprintf(stderr, "Failed to find continuation tree\n");
        return 1;
    }

    da_u32_append(out_ids, 101);

    da_s8 parts = str_split(s8_from_parts((char *)text, text_len), s8_init(" "));
    for (int i = 0; i < parts.len; i++)
    {
        s8 part = parts.data[i];
        int depth = 0;
        const trie_t *node = trie_longest(&tok.trie, part.data, part.len, &depth);
        if (node == NULL)
        {
            return 1;
        }
        da_u32_append(out_ids, node->value);

        size_t remaining_len = part.len - depth;
        if (remaining_len == 0)
            continue;

        // now the remaining part of the text
        const trie_t *cont_node = trie_longest(continuation_tree, part.data + depth, remaining_len, &depth);
        if (cont_node == NULL)
        {
            return 1;
        }
        da_u32_append(out_ids, cont_node->value);
    }
    da_s8_free(&parts);
    da_u32_append(out_ids, 102);
    return 0;
}

void tokenizer_destroy(tokenizer_t *tok)
{
    trie_destroy(&tok->trie);
}