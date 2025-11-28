#include "str.h"
#include "s8.h"

bool str_ends_with(s8 str, char *suffix)
{
    s8 suffix_str = s8_init(suffix);
    if (str.len < suffix_str.len)
    {
        return false;
    }
    s8 slice = s8_slice(str, str.len - suffix_str.len, str.len);
    return s8_eq(slice, suffix_str);
}

size_t str_find_next(s8 str, s8 delim, size_t start)
{
    for (int i = start; i < str.len; i++)
    {
        for (int j = 0; j < delim.len; j++)
        {
            if (str.data[i] == delim.data[j])
                return i;
        }
    }
    return -1;
}

da_s8 str_split(s8 str, s8 delim)
{
    da_s8 result = {0};
    int i = 0;
    for (int i = 0; i < str.len; i++)
    {
        size_t next = str_find_next(str, delim, i);
        if (next == -1)
        {
            da_s8_append(&result, s8_slice(str, i, str.len));
            break;
        }
        s8 slice = s8_slice(str, i, next);
        da_s8_append(&result, slice);
        i = next;
    }
    return result;
}