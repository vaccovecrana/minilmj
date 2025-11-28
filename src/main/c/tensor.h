#pragma once
#include <stddef.h>
#include <stdbool.h>
#include "tbf.h"

#define TENSOR_MAX_DIM 4

// lifecycle
tensor_t tensor_copy(tensor_t t);
tensor_t tensor_create(uint32_t ndim, uint32_t *dims);
void tensor_destroy(tensor_t *t);
tensor_t tensor_view(uint8_t ndim, const uint32_t *dims, float *base_data);
t_status tensor_permute(tensor_t *out, const tensor_t in, uint8_t d0, uint8_t d1);
tensor_t tensor_slice(const tensor_t *t, int dim, uint64_t idx, bool keepdim);

/// @brief 2d matmul: C[M, N] = A[M, K] x B[K, N]
t_status tensor_matmul(tensor_t *out, const tensor_t A, const tensor_t B);

/// @brief Binary operation type
typedef enum bop_t bop_t;

/// Performs element-wise binary operation between two tensors **in-place**
///
/// @param out    Output tensor (modified in-place)
/// @param other  Second operand tensor
/// @param op     Binary operation: B_ADD, B_SUB, B_MUL, B_DIV
/// @return       T_OK on success, T_ERR if tensors have different sizes
///
/// Both tensors must have the same number of elements.
void tensor_binary_op(tensor_t out, const tensor_t other, bop_t op);

/// @brief Supported binary operations
typedef enum bop_t
{
    B_ADD,
    B_SUB,
    B_MUL,
    B_DIV,
    BOP_COUNT
} bop_t;

/// @brief Unary operation type
typedef enum uop_t uop_t;

/// Performs element-wise unary operation on a tensor **in-place**
///
/// @param a        Input tensor
/// @param op       Unary operation: U_NEG, U_EXP, U_LOG, U_GELU, U_ABS, U_SCALE, U_SUB, U_POW
/// @param op_param Optional parameter for the operation (e.g., power for U_POW)
/// @return         T_OK on success, T_ERR if operation fails
void tensor_unary_op(tensor_t a, uop_t op, float *op_param);

/// @brief Supported unary operations
typedef enum uop_t
{
    U_NEG,
    U_EXP,
    U_LOG,
    U_GELU,
    U_ABS,
    U_SCALE,
    U_SUB,
    U_POW,
    UOP_COUNT
} uop_t;

/// Computes the sum of all elements in a tensor.
///
float tensor_sum(tensor_t t);

// debug
void tensor_print(const tensor_t t);
size_t tensor_numel(const tensor_t t);
void tensor_dump(FILE *fp, const tensor_t *t);
void tensor_load(FILE *fp, tensor_t *t);

#define TENSOR_CRASH_ON_ERROR 1
#ifdef TENSOR_CRASH_ON_ERROR
#define M_ABORT_ON_ERROR 1
#else
#define M_ABORT_ON_ERROR 0
#endif

#define m_try(expr)                                     \
    do                                                  \
    {                                                   \
        t_status _res__ = (expr);                       \
        if (_res__ != T_OK)                             \
        {                                               \
            fprintf(stderr, "%s:%d: error in %s: %d\n", \
                    __FILE__, __LINE__, #expr, _res__); \
            if (M_ABORT_ON_ERROR)                       \
            {                                           \
                abort();                                \
            }                                           \
            else                                        \
            {                                           \
                return _res__;                          \
            }                                           \
        }                                               \
    } while (0)
