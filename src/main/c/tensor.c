// tensor.c — tiny, contiguous-only implementation
#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>
#if defined(__linux__) || defined(__GLIBC__)
#include <alloca.h>
#elif defined(__APPLE__)
// alloca is available through stdlib.h on macOS
#endif

static uint32_t prod(const uint32_t *a, size_t n)
{
    uint32_t p = 1;
    for (size_t i = 0; i < n; i++)
        p *= a[i];
    return p;
}

t_status tensor_matmul(tensor_t *out, const tensor_t A, const tensor_t B)
{
    if (A.ndim != 2 || B.ndim != 2)
        return -1;
    uint32_t M = A.dims[0], K = A.dims[1], N = B.dims[1];

    // Require contiguous layout (row-major).
    if (!(A.strides[0] == K && A.strides[1] == 1 &&
          B.strides[0] == N && B.strides[1] == 1))
        return -2;

    *out = tensor_create(2, (uint32_t[]){M, N});
    if (!(out->strides[0] == N && out->strides[1] == 1))
        return -2;

    const float *__restrict a = A.data; // [M,K]
    const float *__restrict b = B.data; // [K,N]
    float *__restrict c = out->data;    // [M,N]

    // Tune these for your cache/CPU. 128/128/64 is often good on modern x86; try 64s on smaller caches.
    const size_t BM = 128, BN = 128, BK = 64;

    // If you want multi-core, enable the pragma below and compile with -fopenmp.
    // Parallelizing by (i0,j0) tiles avoids write races on C.
    for (size_t i0 = 0; i0 < M; i0 += BM)
        for (size_t j0 = 0; j0 < N; j0 += BN)
        {
            const size_t imax = (i0 + BM < M) ? (i0 + BM) : M;
            const size_t jmax = (j0 + BN < N) ? (j0 + BN) : N;

            for (size_t k0 = 0; k0 < K; k0 += BK)
            {
                const size_t kmax = (k0 + BK < K) ? (k0 + BK) : K;

                for (size_t i = i0; i < imax; ++i)
                {
                    float *__restrict c_row = c + i * (size_t)N + j0;

                    for (size_t k = k0; k < kmax; ++k)
                    {
                        const float aik = a[i * (size_t)K + k];
                        const float *__restrict b_row = b + k * (size_t)N + j0;

                        // Inner j loop streams B and C rows (contiguous), good for caches & SIMD.
                        size_t j = j0;
                        // Unroll by 4 (simple, helps auto-vectorization)
                        for (; j + 4 <= jmax; j += 4)
                        {
                            const size_t jj = j - j0;
                            c_row[jj + 0] += aik * b_row[0];
                            c_row[jj + 1] += aik * b_row[1];
                            c_row[jj + 2] += aik * b_row[2];
                            c_row[jj + 3] += aik * b_row[3];
                            b_row += 4;
                        }
                        for (; j < jmax; ++j)
                        {
                            c_row[j - j0] += aik * (*b_row++);
                        }
                    }
                }
            }
        }

    return 0;
}
void tensor_print(const tensor_t t)
{
    const size_t max_decimals = 4;
    const size_t max_print = 3;
    // only print the last 2 dimensions up to the first 10 elements taking the stride into account
    // print dims
    printf("[");
    for (size_t j = 0; j < t.ndim; j++)
        printf("%d ", t.dims[j]);
    printf("]\n");
    for (size_t i = 0; i < t.dims[t.ndim - 2] && i < max_print; i++)
    {
        printf("|");
        for (size_t j = 0; j < t.dims[t.ndim - 1] && j < max_print; j++)
        {
            printf("%*.*f ", (int)(max_decimals + 4), (int)max_decimals, ((float *)t.data)[i * t.strides[t.ndim - 2] + j]);
        }
        printf("|");
        printf("\n");
    }

    printf("\n");
}

tensor_t tensor_create(uint32_t ndim, uint32_t *dims)
{
    tensor_t out = {0};
    memcpy(out.dims, dims, ndim * sizeof(uint32_t));
    out.ndim = ndim;
    out.data = (float *)calloc(prod(dims, ndim), sizeof(float));
    assert(out.data);
    uint64_t s = 1;
    for (int i = ndim - 1; i >= 0; --i)
    {
        out.strides[i] = s;
        s *= dims[i];
    }
    return out;
}

size_t tensor_numel(const tensor_t t)
{
    if (!t.data)
        return 0;
    return prod(t.dims, t.ndim);
}
tensor_t tensor_view(uint8_t ndim, const uint32_t *dims, float *data)
{
    uint64_t strides[ndim];
    uint64_t s = 1;
    for (int i = ndim - 1; i >= 0; --i)
    {
        strides[i] = s;
        s *= dims[i];
    }
    tensor_t t = {0};
    t.dtype = 1;
    t.ndim = ndim;
    memcpy(t.dims, dims, sizeof(uint32_t) * ndim);
    memcpy(t.strides, strides, sizeof(uint64_t) * ndim);
    t.data = data;
    return t;
}

tensor_t tensor_slice(const tensor_t *t, int dim, uint64_t idx, bool keepdim)
{
    // normalize dim (allow negative)
    uint8_t ndim = t->ndim;
    if (dim < 0)
        dim += ndim;
    // bounds check - return zero tensor instead of crashing
    if (dim < 0 || dim >= ndim || idx >= t->dims[dim])
    {
        return (tensor_t){0};
    }

    // compute new base pointer
    float *base = t->data + idx * t->strides[dim];

    if (keepdim)
    {
        // same rank; dims[dim]=1, strides can stay (or 0 if you prefer)
        uint32_t *dims = alloca(sizeof(uint32_t) * ndim);
        uint64_t *strides = alloca(sizeof(uint64_t) * ndim);
        for (int i = 0; i < ndim; ++i)
        {
            dims[i] = t->dims[i];
            strides[i] = t->strides[i];
        }
        dims[dim] = 1;
        return tensor_view(ndim, dims, base);
    }
    else
    {
        // drop that axis
        uint32_t *dims = alloca(sizeof(uint32_t) * (ndim - 1));
        uint64_t *strides = alloca(sizeof(uint64_t) * (ndim - 1));
        for (int i = 0, j = 0; i < ndim; ++i)
        {
            if (i == dim)
                continue;
            dims[j] = t->dims[i];
            strides[j] = t->strides[i];
            ++j;
        }
        return tensor_view(ndim - 1, dims, base);
    }
}

// Assumes float data; adapt T if needed.
void tensor_permute_(tensor_t out, const tensor_t in, const uint8_t *perm)
{
    const uint8_t nd = in.ndim;

    // --- Fast sanity (optional) ---
    // same ndim, dims agree as perm(in)
    // ...

    // Precompute: input stride seen from each OUT dimension.
    // in_step_for_out[k] == stride in 'in' when OUT's k-th coordinate increments by 1
    uint64_t in_step_for_out[TENSOR_MAX_DIM];
    for (uint8_t k = 0; k < nd; ++k)
        in_step_for_out[k] = in.strides[perm[k]];

    // --- Fast path 1: total identity ---
    bool identity = true;
    for (uint8_t k = 0; k < nd; ++k)
        identity &= (perm[k] == k);
    if (identity)
    {
        // If layouts are identical and contiguous, just memcpy.
        const uint64_t numel = tensor_numel(in);
        memcpy(out.data, in.data, numel * sizeof(float));
        return;
    }

    // --- Fast path 2: batched memcpy (unit-stride inner on both sides) ---
    // Choose inner dimension (last in OUT order).
    const uint8_t inner = nd ? (uint8_t)(nd - 1) : 0;
    const bool out_unit = (out.strides[inner] == 1);
    const bool in_unit = (in_step_for_out[inner] == 1);

    if (out_unit && in_unit)
    {
        // We can copy contiguous runs of length = out.dims[inner].
        const uint64_t run = out.dims[inner]; // elements per memcpy
        const size_t bytes = (size_t)run * sizeof(float);

        // Build an odometer over the OUT dims except the inner.
        uint32_t coord[TENSOR_MAX_DIM] = {0};
        uint64_t in_off = 0;
        uint64_t out_off = 0;

        // Precompute rollback (total delta when a dim wraps)
        uint64_t in_step[TENSOR_MAX_DIM], out_step[TENSOR_MAX_DIM];
        for (uint8_t k = 0; k < nd; ++k)
        {
            in_step[k] = in_step_for_out[k];
            out_step[k] = out.strides[k];
        }
        // Inner-step deltas:
        const uint64_t in_inner_step = in_step[inner];
        const uint64_t out_inner_step = out_step[inner]; // == 1

        // Process all runs
        const uint64_t total_runs = (nd == 0) ? 1 : ({ uint64_t tr = 1; for (uint8_t k=0;k<nd-1;++k) tr *= out.dims[k]; tr; });

        for (uint64_t r = 0; r < total_runs; ++r)
        {
            // copy the full inner line
            memcpy(out.data + out_off, in.data + in_off, bytes);

            // advance the odometer over dims [0..nd-2]
            int k = (int)nd - 2;
            for (; k >= 0; --k)
            {
                coord[k]++;
                in_off += in_step[k];
                out_off += out_step[k];
                if (coord[k] < out.dims[k])
                    break;

                // wrap dim k
                coord[k] = 0;
                in_off -= (uint64_t)out.dims[k] * in_step[k];
                out_off -= (uint64_t)out.dims[k] * out_step[k];
            }
            (void)in_inner_step;
            (void)out_inner_step; // silence unused if nd==1
        }
        return;
    }

    // --- Generic, division-free odometer (one element at a time) ---
    uint32_t coord[TENSOR_MAX_DIM] = {0};

    uint64_t in_off = 0;
    uint64_t out_off = 0;

    // Precompute steps
    uint64_t in_step[TENSOR_MAX_DIM], out_step[TENSOR_MAX_DIM];
    for (uint8_t k = 0; k < nd; ++k)
    {
        in_step[k] = in_step_for_out[k];
        out_step[k] = out.strides[k];
    }

    // Total elements
    uint64_t numel = 1;
    for (uint8_t k = 0; k < nd; ++k)
        numel *= out.dims[k];

    for (uint64_t cnt = 0; cnt < numel; ++cnt)
    {
        out.data[out_off] = in.data[in_off];

        // Increment last (innermost OUT) coordinate; cascade carries.
        int k = (int)nd - 1;
        for (; k >= 0; --k)
        {
            coord[k]++;
            in_off += in_step[k];
            out_off += out_step[k];
            if (coord[k] < out.dims[k])
                break;

            // wrap dim k
            coord[k] = 0;
            in_off -= (uint64_t)out.dims[k] * in_step[k];
            out_off -= (uint64_t)out.dims[k] * out_step[k];
        }
        // when k < 0 we are done on next loop exit
    }
}

#define m_swap(a, b)              \
    do                            \
    {                             \
        __typeof__(a) _tmp = (a); \
        (a) = (b);                \
        (b) = _tmp;               \
    } while (0)

t_status tensor_permute(tensor_t *out, const tensor_t in, uint8_t d0, uint8_t d1)
{
    uint8_t perm[TENSOR_MAX_DIM] = {0, 1, 2, 3};
    const uint8_t nd = in.ndim;
    if (sizeof(perm) < TENSOR_MAX_DIM)
        return T_ERR;
    if (nd > TENSOR_MAX_DIM)
        return T_ERR;
    if (d0 >= nd || d1 >= nd)
        return T_ERR;
    m_swap(perm[d0], perm[d1]);
    // tensor_t tmp = tensor_tmp_permuted(in, d0, d1);
    uint32_t dims[TENSOR_MAX_DIM];
    memcpy(dims, in.dims, sizeof(uint32_t) * in.ndim);
    m_swap(dims[d0], dims[d1]);
    *out = tensor_create(in.ndim, dims);
    tensor_permute_(*out, in, perm);
    return T_OK;
}

float tensor_sum(tensor_t t)
{
    float sum = 0.0f;
    for (size_t i = 0; i < prod(t.dims, t.ndim); i++)
        sum += t.data[i];
    return sum;
}

tensor_t tensor_copy(tensor_t t)
{
    tensor_t out = tensor_create(t.ndim, t.dims);
    memcpy(out.data, t.data, prod(t.dims, t.ndim) * sizeof(float));
    return out;
}

void tensor_dump(FILE *fp, const tensor_t *t)
{
    // serialize the tensor to a file
    uint64_t nbytes = prod(t->dims, t->ndim);

    fwrite(&t->ndim, sizeof(uint8_t), 1, fp);
    fwrite(t->dims, sizeof(uint32_t), t->ndim, fp);
    fwrite(&nbytes, sizeof(uint64_t), 1, fp);
    fwrite(t->strides, sizeof(uint64_t), t->ndim, fp);
    fwrite(t->data, sizeof(float), nbytes, fp);
}

void tensor_load(FILE *fp, tensor_t *t)
{

    uint8_t ndim;
    uint32_t dims[TENSOR_MAX_DIM];
    uint64_t nbytes;
    uint64_t strides[TENSOR_MAX_DIM];

    fread(&ndim, sizeof(uint8_t), 1, fp);
    fread(dims, sizeof(uint32_t), ndim, fp);
    fread(&nbytes, sizeof(uint64_t), 1, fp);
    fread(strides, sizeof(uint64_t), ndim, fp);

    *t = tensor_create(ndim, dims);

    fread(t->data, sizeof(float), nbytes, fp);
}

void tensor_destroy(tensor_t *t)
{
    free(t->data);
    t->data = NULL;
    for (int i = 0; i < t->ndim; i++)
    {
        t->dims[i] = 0;
        t->strides[i] = 0;
    }
    t->ndim = 0;
}

static inline float op_neg(float x, float *op_param) { return -x; }
static inline float op_exp(float x, float *op_param) { return expf(x); }
static inline float op_log(float x, float *op_param) { return logf(x); }
static inline float op_abs(float x, float *op_param) { return fabsf(x); }
static inline float op_scale(float x, float *op_param) { return x * op_param[0]; }
static inline float op_pow(float x, float *op_param) { return powf(x, op_param[0]); }
static inline float op_sub(float x, float *op_param) { return x - op_param[0]; }
static inline float op_gelu(float x, float *op_param)
{
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * powf(x, 3.0f))));
}
typedef float (*unary_op_t)(float x, float *op_param);

static unary_op_t unary_op_table[UOP_COUNT] = {op_neg, op_exp, op_log, op_gelu, op_abs, op_scale, op_sub, op_pow};

void tensor_unary_op(tensor_t a, uop_t op, float *op_param)
{
    assert(op < UOP_COUNT);
    const uint64_t N = tensor_numel(a);
    unary_op_t op_func = unary_op_table[op];
    float *x = a.data; // [N]

    for (uint64_t i = 0; i < N; i++)
        x[i] = op_func(x[i], op_param);
}

static float binop_addf(float x, float y) { return x + y; }
static float binop_subf(float x, float y) { return x - y; }
static float binop_mulf(float x, float y) { return x * y; }
static float binop_divf(float x, float y) { return x / y; }
typedef float (*bop_fn)(float, float);
static bop_fn bop_table[BOP_COUNT] = {binop_addf, binop_subf, binop_mulf, binop_divf};

static inline bool is_lastdim_bias(const tensor_t out, const tensor_t other)
{
    // check if first n-1 dims are single element
    for (uint64_t i = 0; i < other.ndim - 1; i++)
    {
        if (other.dims[i] != 1)
            return false;
    }
    // and the last dim is the same as the other dims
    if (other.dims[other.ndim - 1] != out.dims[out.ndim - 1])
        return false;
    return true;
}

static inline bool is_same_shape(const tensor_t out, const tensor_t other)
{
    for (uint64_t i = 0; i < out.ndim; i++)
    {
        if (out.dims[i] != other.dims[i])
            return false;
    }
    return true;
}

void tensor_binary_broadcast(tensor_t out, const tensor_t other, bop_t op)
{
    // bias on last dimension: out:[..., N], other:[N]
    assert(is_lastdim_bias(out, other));
    bop_fn op_func = bop_table[op];
    uint64_t n_out = tensor_numel(out);
    uint64_t n_other = tensor_numel(other);

    const uint64_t N = other.dims[0];
    const uint64_t rows = (N ? n_out / N : 0);
    float *__restrict y = out.data;
    const float *__restrict b = other.data;

    for (uint64_t r = 0; r < rows; ++r)
    {
        uint64_t base = r * N;
        for (uint64_t j = 0; j < N; ++j)
            y[base + j] = op_func(y[base + j], b[j]);
    }
}

void tensor_binary_op(tensor_t out, const tensor_t other, bop_t op)
{
    bool can_broadcast = is_lastdim_bias(out, other);
    bool same_shape = is_same_shape(out, other);

    if (same_shape)
    {
        bop_fn op_func = bop_table[op];
        for (uint64_t i = 0; i < tensor_numel(out); i++)
            out.data[i] = op_func(out.data[i], other.data[i]);
        return;
    }
    else if (can_broadcast)
    {
        tensor_binary_broadcast(out, other, op);
        return;
    }
    assert(false);
}
