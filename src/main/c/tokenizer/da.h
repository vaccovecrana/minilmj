#pragma once
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include "types.h"
#define DA_INITIAL_CAPACITY 512

#define DA_GROWTH_FACTOR 1.5

typedef struct da
{
    struct da *storage;
    size_t iter_index;
    size_t len;
    size_t capacity;
    void *data;
} da;

static inline void _da_grow(da *array, size_t item_size, size_t extra_size)
{
    if (array->len >= array->capacity || extra_size > (array->capacity - array->len))
    {
        size_t new_capacity =
            array->capacity == 0
                ? DA_INITIAL_CAPACITY
                : (size_t)(DA_GROWTH_FACTOR * (float)(array->capacity));
        if (new_capacity < (extra_size + array->len))
        {
            new_capacity = extra_size + array->len;
        }
        array->data = realloc(array->data, new_capacity * item_size);
        array->capacity = new_capacity;
    }
}

static inline void _da_remove(da *array, size_t index)
{
    memmove(array->data + index, array->data + index + 1, array->len - index - 1);
    array->len--;
}

static inline void _da_free(da *array)
{
    free(array->data);
    if (array->storage)
    {
        _da_free(array->storage);
    }
    array->data = NULL;
    array->len = 0;
    array->capacity = 0;
}

static inline void *_da_extend(da *array, void *data, size_t len, size_t item_size)
{
    size_t total_size = item_size * len;
    _da_grow(array, item_size, total_size);
    memcpy(array->data + array->len * item_size, data, total_size);

    void *ptr = array->data + array->len * item_size;
    array->len += len;
    return ptr;
}

static inline void da_reset(da *array)
{
    array->iter_index = 0;
    array->len = 0;
}

static inline bool da_eq(da *a, da *b)
{
    if (a->len != b->len)
    {
        return false;
    }
    return memcmp(a->data, b->data, a->len) == 0;
}

#define DA(type)                                                                      \
    typedef struct da_##type                                                          \
    {                                                                                 \
        da *storage;                                                                  \
        size_t iter_index;                                                            \
        size_t len;                                                                   \
        size_t capacity;                                                              \
        type *data;                                                                   \
    } da_##type;                                                                      \
    static inline void da_##type##_append(da_##type *array, type element)             \
    {                                                                                 \
        _da_grow((da *)array, sizeof(type), 1);                                       \
        array->data[array->len++] = element;                                          \
    }                                                                                 \
    static inline void da_##type##_free(da_##type *array)                             \
    {                                                                                 \
        _da_free((da *)array);                                                        \
    }                                                                                 \
    static inline void da_##type##_remove(da_##type *array, size_t index)             \
    {                                                                                 \
        _da_remove((da *)array, index);                                               \
    }                                                                                 \
    static inline void *da_##type##_extend(da_##type *array, type *data, size_t size) \
    {                                                                                 \
        return _da_extend((da *)array, data, size, sizeof(type));                     \
    }                                                                                 \
    static inline bool da_##type##_eq(da_##type *a, da_##type *b)                     \
    {                                                                                 \
        return da_eq((da *)a, (da *)b);                                               \
    }                                                                                 \
    static inline type *da_##type##_next(da_##type *array)                            \
    {                                                                                 \
        if (array->iter_index >= array->len)                                          \
        {                                                                             \
            array->iter_index = 0;                                                    \
            return NULL;                                                              \
        }                                                                             \
        return &array->data[array->iter_index++];                                     \
    }                                                                                 \
    static inline void da_##type##_reset(da_##type *array)                            \
    {                                                                                 \
        array->iter_index = 0;                                                        \
        array->len = 0;                                                               \
    }
