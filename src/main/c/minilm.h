#pragma once

#include <stddef.h>
#include "tensor.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/s8.h"
#include "tokenizer/da.h"

// Maximum number of tokens supported by the model
static const size_t MINILM_MAX_TOKENS = 256;

typedef struct minilm_t minilm_t;

/// @brief Load weights from tbf file and initialize the tokenizer using vocab.txt
/// @param m minilm_t
/// @param tbf_path path to tbf file
/// @param vocab_txt_path path to vocab.txt
/// @return 0 on success, 1 on error
int minilm_create(minilm_t *m, const char *tbf_path, const char *vocab_txt_path);

/// @brief Embed a string into a tensor of token ids
/// Internally calls minilm_tokenize and minilm_encode
/// @param m minilm_t
/// @param str string to embed
/// @param str_len length of the string
/// @return tensor of token ids
t_status minilm_embed(minilm_t m, char *str, size_t str_len, tensor_t *out);

/// @brief Destroy the minilm_t and free the memory
void minilm_destroy(minilm_t *m);

/// @brief Encode a string into a tensor of token ids
t_status minilm_encode(minilm_t m, da_u32 ids, tensor_t *out);

/// @brief Tokenize a string into a tensor of token ids
t_status minilm_tokenize(minilm_t m, s8 str, da_u32 *ids);

/// PyTorch reference:
/// ```python
// BertModel(
//     (embeddings): BertEmbeddings(
//       (word_embeddings): Embedding(30522, 384, padding_idx=0)
//       (position_embeddings): Embedding(512, 384)
//       (token_type_embeddings): Embedding(2, 384)
//       (LayerNorm): LayerNorm((384,), eps=1e-12, elementwise_affine=True)
//       (dropout): Dropout(p=0.1, inplace=False)
//     )
//     (encoder): BertEncoder(
//       (layer): ModuleList(
//         (0-5): 6 x BertLayer(
//           (attention): BertAttention(
//             (self): BertSdpaSelfAttention(
//               (query): Linear(in_features=384, out_features=384, bias=True)
//               (key): Linear(in_features=384, out_features=384, bias=True)
//               (value): Linear(in_features=384, out_features=384, bias=True)
//               (dropout): Dropout(p=0.1, inplace=False)
//             )
//             (output): BertSelfOutput(
//               (dense): Linear(in_features=384, out_features=384, bias=True)
//               (LayerNorm): LayerNorm((384,), eps=1e-12,
//               elementwise_affine=True) (dropout): Dropout(p=0.1,
//               inplace=False)
//             )
//           )
//           (intermediate): BertIntermediate(
//             (dense): Linear(in_features=384, out_features=1536, bias=True)
//             (intermediate_act_fn): GELUActivation()
//           )
//           (output): BertOutput(
//             (dense): Linear(in_features=1536, out_features=384, bias=True)
//             (LayerNorm): LayerNorm((384,), eps=1e-12,
//             elementwise_affine=True) (dropout): Dropout(p=0.1, inplace=False)
//           )
//         )
//       )
//     )
//     (pooler): BertPooler(
//       (dense): Linear(in_features=384, out_features=384, bias=True)
//       (activation): Tanh()
//     )
//   )
/// ```

// structs to store weights
struct output_layer_t
{
  tensor_t weight;   // [HIDDEN_SIZE, HIDDEN_SIZE]
  tensor_t bias;     // [1, HIDDEN_SIZE]
  tensor_t ln_gamma; // [1, HIDDEN_SIZE]
  tensor_t ln_beta;  // [1, HIDDEN_SIZE]
};
typedef struct bert_layer_weigts_t
{
  tensor_t query;      // [HIDDEN_SIZE, HIDDEN_SIZE]
  tensor_t query_bias; // [1, HIDDEN_SIZE]
  tensor_t key;        // [HIDDEN_SIZE, HIDDEN_SIZE]
  tensor_t key_bias;   // [1, HIDDEN_SIZE]
  tensor_t value;      // [HIDDEN_SIZE, HIDDEN_SIZE]
  tensor_t value_bias; // [1, HIDDEN_SIZE]
  // output
  struct output_layer_t output;
  struct intermediate
  {
    tensor_t weight; // [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    tensor_t bias;   // [1, INTERMEDIATE_SIZE]
  } intermediate;

  struct output_layer_t output_2;
} bert_layer_weigts_t;

typedef struct minilm_t
{
  TbfFile tf;
  tokenizer_t tokenizer;
  // embeddings
  struct embeddings
  {
    tensor_t word;     // [VOCAB_SIZE, HIDDEN_SIZE]
    tensor_t pos;      // [MAX_POS,   HIDDEN_SIZE]
    tensor_t type;     // [TOKEN_TYPES, HIDDEN_SIZE]
    tensor_t ln_gamma; // [1, HIDDEN_SIZE]
    tensor_t ln_beta;  // [1, HIDDEN_SIZE]
  } embeddings;

  // encoder
  //  attention
  bert_layer_weigts_t attention[6]; // 6 layers
  // intermediate
  tensor_t intermediate_weight; // [HIDDEN_SIZE, HIDDEN_SIZE]
  tensor_t intermediate_bias;   // [1, HIDDEN_SIZE]
  // output
  tensor_t output_weight;   // [HIDDEN_SIZE, HIDDEN_SIZE]
  tensor_t output_bias;     // [1, HIDDEN_SIZE]
  tensor_t output_ln_gamma; // [1, HIDDEN_SIZE]
  tensor_t output_ln_beta;  // [1, HIDDEN_SIZE]
} minilm_t;
