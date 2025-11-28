#pragma once

#include <stddef.h>
#include <stdint.h>

typedef struct trie_t
{                     /* Node with inline child arrays */
  uint32_t value;     /* -1 if non-terminal */
  uint8_t key;        /* only works for 1 byte for now */
  uint8_t label[100]; // debug only
  struct trie_t *children;
  size_t length;
  size_t capacity;
} trie_t;

int trie_create(trie_t t);
int trie_destroy(trie_t *t);

// Insert a token string (as in vocab.txt) with its vocab ID
int trie_insert(trie_t *t, const uint8_t *token, uint32_t value);

// Step one byte from a node, return child node index or -1 if no edge
trie_t *trie_find_child(trie_t n, uint8_t b);

// Greedy longest match starting from `start_node`
//   s      = pointer to bytes
//   len    = number of bytes
//   out_id = best token ID found (-1 if none)
//   out_len= length of that best match in bytes
// Returns 1 if any match found, 0 if no match
const trie_t *trie_longest(const trie_t t, const uint8_t *s, int len, int *offset);

// Debugging functions
void trie_dump(trie_t t);
void trie_dump_tree(trie_t t, int indent);
