#pragma once
 
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define TENSOR_MAX_DIM 4
#define TENSOR_MAX_NAME_LEN 128

typedef enum
{
  T_OK = 0,
  T_ERR,
  T_TOKEN_LIMIT_EXCEEDED
} t_status;

// DTYPE_MAP = {
//   torch.float32 : 1,
//   torch.float16 : 2,
//   torch.float64 : 3,
//   torch.int64 : 4,
//   torch.int32 : 5,
//   torch.uint8 : 6,
// }

typedef struct
{
  uint64_t offset;

  char name[TENSOR_MAX_NAME_LEN]; // name of the tensor
  uint8_t dtype;                  // dtype of the tensor (1: float32, 2: float16, 3: float64, 4: int64, 5: int32, 6: uint8)
  uint8_t ndim;
  uint32_t dims[TENSOR_MAX_DIM];
  uint64_t nbytes;
  float *data;
  uint64_t strides[TENSOR_MAX_DIM];
} tensor_t;

typedef struct
{
  FILE *fp;
  uint64_t count;
  tensor_t *tensors;
} TbfFile;

t_status tbf_open(TbfFile *tf, const char *path);
tensor_t *tbf_get_tensor(TbfFile tf, const char *name);
void tbf_close(TbfFile f);
void tbf_print_tensors(TbfFile tf);
