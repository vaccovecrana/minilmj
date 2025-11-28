#pragma once
#include "s8.h"

bool str_ends_with(s8 str, char *suffix);

size_t str_find_next(s8 str, s8 delim, size_t start);

da_s8 str_split(s8 str, s8 delim);