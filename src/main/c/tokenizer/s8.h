#pragma once
#include "types.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "da.h"
#define m_s8(cstr) \
    (s8) { .data = (u8 *)cstr, .len = _strlen(cstr) }

DA(u8)
typedef da_u8 s8;

s8 s8_init(char *cstr);
s8 s8_from_parts(char *cstr, size_t len);
void s8_free(s8 *s);
const char *s8_cstr(const s8 s);
bool s8_eq(s8 s1, s8 s2);
bool s8_eq_cstr(s8 s, char *cstr);
s8 s8_slice(s8 s, size_t start, size_t end);

DA(s8);
void da_s8_append_many(da_s8 *array, char *first, ...);

size_t _strlen(char *cstr);
