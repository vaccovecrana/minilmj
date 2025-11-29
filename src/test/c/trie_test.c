#include "trie.h"
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>

int main()
{
    trie_t t = {0};
    trie_insert(&t, (uint8_t *)"hello", 1037);
    trie_insert(&t, (uint8_t *)"hella", 1038);
    const char *s = "hella";
    int offset;
    const trie_t *node = trie_longest(&t, (uint8_t *)s, strlen(s), &offset);
    assert(node->value == 1038);
    s = "hello";
    node = trie_longest(&t, (uint8_t *)s, strlen(s), &offset);
    assert(node->value == 1037);
    s = "world";
    node = trie_longest(&t, (uint8_t *)s, strlen(s), &offset);
    assert(node->value == 0);
    trie_destroy(&t);
}