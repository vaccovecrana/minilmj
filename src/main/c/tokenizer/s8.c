#include "s8.h"

s8 s8_init(char *cstr)
{
    return (s8){.data = (u8 *)cstr, .len = _strlen(cstr)};
}

s8 s8_from_parts(char *cstr, size_t len)
{
    return (s8){.data = (u8 *)cstr, .len = len};
}

size_t _strlen(char *cstr)
{
    if (!cstr)
    {
        return 0;
    }
    return strlen(cstr);
}

const char *s8_cstr(const s8 s) { return (const char *)s.data; }

bool s8_eq(s8 s1, s8 s2)
{
    return da_u8_eq(&s1, &s2);
}

bool s8_eq_cstr(s8 s, char *cstr)
{
    return s8_eq(s, s8_init(cstr));
}

s8 s8_slice(s8 s, size_t start, size_t end)
{
    return (s8){.data = s.data + start, .len = end - start};
}

void da_s8_append_many(da_s8 *array, char *first, ...)
{
    va_list args;
    va_start(args, first);
    while (1)
    {
        char *element = va_arg(args, char *);
        if (element == NULL)
            break;
        da_s8_append(array, s8_init(element));
    }
    va_end(args);
}