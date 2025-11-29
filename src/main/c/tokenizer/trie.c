#include "trie.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define max(a, b) ((a) > (b) ? (a) : (b))

trie_t *trie_find_child(const trie_t *n, uint8_t b)
{
    for (int i = 0; i < n->length; i++)
        if (n->children[i].key == b)
        {
            return &n->children[i];
        }
    return NULL;
}

static trie_t *trie_add_child(trie_t *n, uint8_t b)
{
    trie_t child = {.key = b};
    if (n->length == n->capacity)
    {
        // bug on realloc
        n->capacity = max(8, n->capacity * 2);
        n->children = realloc(n->children, n->capacity * sizeof(trie_t));
        if (n->children == NULL)
        {
            fprintf(stderr, "trie_add_child: realloc failed\n");
            return NULL;
        }
    }
    n->children[n->length] = child;
    n->length++;
    return &n->children[n->length - 1];
}

int trie_insert(trie_t *t, const uint8_t *token, uint32_t value)
{
    trie_t *node = t;
    for (int i = 0; token[i] != '\0'; ++i)
    {
        uint8_t b = token[i];
        trie_t *child = trie_find_child(node, b);
        if (child == NULL)
        {
            child = trie_add_child(node, b);
            if (child == NULL)
                return 1;
        }
        node = child;
    }
    node->value = value;
    strncpy((char *)node->label, (char *)token, sizeof(node->label));
    return 0;
}

const trie_t *trie_longest(const trie_t *t, const uint8_t *s, int len, int *offset)
{
    const trie_t *node = t;
    *offset = 0;
    for (int i = 0; i < len; ++i)
    {
        uint8_t b = s[i];
        trie_t *child = trie_find_child(node, b);

        if (child == NULL)
            break;

        node = child;
        *offset = i + 1;
    }

    return node;
}
int trie_create(trie_t t) { return 0; }
int trie_destroy(trie_t *t)
{
    for (int i = 0; i < t->length; i++)
    {
        trie_destroy(&t->children[i]);
    }
    free(t->children);

    t->children = NULL;
    t->length = 0;
    t->capacity = 0;
    return 0;
}

void trie_dump_tree(trie_t t, int indent)
{
    printf("%*strie_t {\n", indent, "");
    printf("%*s  value: %d\n", indent, "", t.value);
    if (t.length == 0)
        printf("%*s  label: %s\n", indent, "", t.label);
    if (t.length)
        printf("%*s  children(%d):\n", indent, "", (int)t.length);
    for (int i = 0; i < t.length; i++)
    {
        trie_dump_tree(t.children[i], indent + 4);
    }

    printf("%*s}\n", indent, "");
}

void trie_dump(trie_t t)
{
    printf("trie_t {\n");
    printf("  value: %d\n", t.value);
    printf("  label: %s\n", t.label);
    if (t.length)
        printf("  children(%d):\n", (int)t.length);
    printf("}\n");
}
