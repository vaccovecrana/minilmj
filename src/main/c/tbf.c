#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "tbf.h"

static int read_exact(void *dst, size_t sz, FILE *fp)
{
    return fread(dst, 1, sz, fp) == sz ? 0 : -1;
}

t_status tbf_open(TbfFile *tf, const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp)
    {
        fprintf(stderr, "Failed to open TBF file\n");
        return T_ERR;
    }

    char magic[4];
    if (read_exact(magic, 4, fp) < 0 || memcmp(magic, "TBF1", 4) != 0)
    {
        fclose(fp);
        return T_ERR;
    }

    uint64_t count;
    if (read_exact(&count, 8, fp) < 0)
    {
        fclose(fp);
        return T_ERR;
    }

    tensor_t *ts = (tensor_t *)calloc(count, sizeof(tensor_t));
    if (!ts)
    {
        fclose(fp);
        return T_ERR;
    }

    for (uint64_t i = 0; i < count; ++i)
    {
        uint16_t name_len;
        if (read_exact(&name_len, 2, fp) < 0)
            goto fail;

        if (read_exact(ts[i].name, name_len, fp) < 0)
            goto fail;
        ts[i].name[name_len] = '\0';

        if (read_exact(&ts[i].dtype, 1, fp) < 0)
            goto fail;
        if (read_exact(&ts[i].ndim, 1, fp) < 0)
            goto fail;

        if (read_exact(ts[i].dims, ts[i].ndim * 4, fp) < 0)
            goto fail;

        if (read_exact(&ts[i].offset, 8, fp) < 0)
            goto fail;
        if (read_exact(&ts[i].nbytes, 8, fp) < 0)
            goto fail;

        long prev_pos = ftell(fp);

        void *buf = malloc(ts[i].nbytes);
        if (!buf)
            goto fail;
        if (fseek(fp, (long)ts[i].offset, SEEK_SET) != 0)
            goto fail;
        if (read_exact(buf, (size_t)ts[i].nbytes, fp) < 0)
            goto fail;
        ts[i].data = buf;
        uint64_t s = 1;
        for (int j = ts[i].ndim - 1; j >= 0; --j)
        {
            ts[i].strides[j] = s;
            s *= ts[i].dims[j];
        }
        fseek(fp, prev_pos, SEEK_SET);
    }

    *tf = (TbfFile){
        .fp = fp,
        .count = count,
        .tensors = ts,
    };
    return T_OK;

fail:
    fprintf(stderr, "Failed to read TBF file\n");
    return T_ERR;
}

static int64_t tbf_find_by_name(TbfFile tf, const char *name)
{
    for (uint64_t i = 0; i < tf.count; ++i)
        if (strcmp(tf.tensors[i].name, name) == 0)
            return (int64_t)i;
    return -1;
}

tensor_t *tbf_get_tensor(TbfFile tf, const char *name)
{
    int64_t idx = tbf_find_by_name(tf, name);
    if (idx < 0)
        return NULL;
    tensor_t *t = &tf.tensors[idx];
    return t;
}

void tbf_close(TbfFile tf)
{
    for (uint64_t i = 0; i < tf.count; ++i)
    {
        if (tf.tensors[i].data)
            free(tf.tensors[i].data);
    }
    if (tf.tensors)
        free(tf.tensors);
    if (tf.fp)
        fclose(tf.fp);
}

void tbf_print_tensors(TbfFile tf)
{
    printf("========================================\n");
    printf("TBF file contains %llu tensors: \n", (unsigned long long)tf.count);
    for (uint64_t i = 0; i < tf.count; ++i)
    {
        printf("%-50s (dtype=%d, ndim=%d, nbytes=%8lu, offset=%lu, shape=(", tf.tensors[i].name, tf.tensors[i].dtype, tf.tensors[i].ndim, (unsigned long)tf.tensors[i].nbytes, (unsigned long)tf.tensors[i].offset);
        for (uint8_t j = 0; j < tf.tensors[i].ndim; ++j)
            printf("%d, ", tf.tensors[i].dims[j]);
        printf("))\n");
    }
    printf("========================================\n");
}
