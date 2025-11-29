#include "minilm.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "tbf.h"
#include "nn.h"
#include "s8.h"
#include "tokenizer.h"

void init_mat_f32(TbfFile tf, const char *name, tensor_t *out)
{
    tensor_t *t = tbf_get_tensor(tf, name);
    if (!t)
    {
        fprintf(stderr, "Failed to get tensor from TBF file\n");
        exit(1);
    }
    *out = *t;
}

t_status minilm_tokenize(minilm_t m, s8 str, da_u32 *ids)
{
    m_try(tokenizer_encode(m.tokenizer, (uint8_t *)str.data, str.len, ids));

    // Check if token count exceeds limit BEFORE padding
    if (ids->len > MINILM_MAX_TOKENS)
    {
        return T_TOKEN_LIMIT_EXCEEDED;
    }

    // pad up to MINILM_MAX_TOKENS
    for (size_t i = ids->len; i < MINILM_MAX_TOKENS; i++)
    {
        da_u32_append(ids, 0);
    }

    return T_OK;
}

void minilm_weights_init(TbfFile tf, minilm_t *weights)
{
    init_mat_f32(tf, "embeddings.word_embeddings.weight", &weights->embeddings.word);
    init_mat_f32(tf, "embeddings.token_type_embeddings.weight", &weights->embeddings.type);
    init_mat_f32(tf, "embeddings.position_embeddings.weight", &weights->embeddings.pos);
    init_mat_f32(tf, "embeddings.LayerNorm.weight", &weights->embeddings.ln_gamma);
    init_mat_f32(tf, "embeddings.LayerNorm.bias", &weights->embeddings.ln_beta);

    // encoder layer
    for (size_t i = 0; i < 6; i++)
    {
        bert_layer_weigts_t *attn = &weights->attention[i];
        char name[100];
        snprintf(name, sizeof(name), "encoder.layer.%zu.attention.self.query.weight", i);
        init_mat_f32(tf, name, &attn->query);
        snprintf(name, sizeof(name), "encoder.layer.%zu.attention.self.query.bias", i);
        init_mat_f32(tf, name, &attn->query_bias);
        snprintf(name, sizeof(name), "encoder.layer.%zu.attention.self.key.weight", i);
        init_mat_f32(tf, name, &attn->key);
        snprintf(name, sizeof(name), "encoder.layer.%zu.attention.self.key.bias", i);
        init_mat_f32(tf, name, &attn->key_bias);
        snprintf(name, sizeof(name), "encoder.layer.%zu.attention.self.value.weight", i);
        init_mat_f32(tf, name, &attn->value);
        snprintf(name, sizeof(name), "encoder.layer.%zu.attention.self.value.bias", i);
        init_mat_f32(tf, name, &attn->value_bias);
        snprintf(name, sizeof(name), "encoder.layer.%zu.attention.output.dense.weight", i);
        init_mat_f32(tf, name, &attn->output.weight);
        snprintf(name, sizeof(name), "encoder.layer.%zu.attention.output.dense.bias", i);
        init_mat_f32(tf, name, &attn->output.bias);
        snprintf(name, sizeof(name), "encoder.layer.%zu.attention.output.LayerNorm.weight", i);
        init_mat_f32(tf, name, &attn->output.ln_gamma);
        snprintf(name, sizeof(name), "encoder.layer.%zu.attention.output.LayerNorm.bias", i);
        init_mat_f32(tf, name, &attn->output.ln_beta);
        snprintf(name, sizeof(name), "encoder.layer.%zu.intermediate.dense.weight", i);
        init_mat_f32(tf, name, &attn->intermediate.weight);
        snprintf(name, sizeof(name), "encoder.layer.%zu.intermediate.dense.bias", i);
        init_mat_f32(tf, name, &attn->intermediate.bias);
        snprintf(name, sizeof(name), "encoder.layer.%zu.output.dense.weight", i);
        init_mat_f32(tf, name, &attn->output_2.weight);
        snprintf(name, sizeof(name), "encoder.layer.%zu.output.dense.bias", i);
        init_mat_f32(tf, name, &attn->output_2.bias);
        snprintf(name, sizeof(name), "encoder.layer.%zu.output.LayerNorm.weight", i);
        init_mat_f32(tf, name, &attn->output_2.ln_gamma);
        snprintf(name, sizeof(name), "encoder.layer.%zu.output.LayerNorm.bias", i);
        init_mat_f32(tf, name, &attn->output_2.ln_beta);
    }
}

tensor_t minilm_embedder_forward(da_u32 ids, minilm_t weights)
{
    tensor_t word_out, pos_out, type_out;
    uint32_t num_tokens = ids.len;
    nn_embeddings_forward(&word_out, ids.data, num_tokens, weights.embeddings.word);

    tensor_t position_ids = tensor_create(1, (uint32_t[]){num_tokens});
    for (size_t i = 0; i < num_tokens; i++)
    {
        ((int *)position_ids.data)[i] = i;
    }
    nn_embeddings_forward(&pos_out, (uint32_t *)position_ids.data, num_tokens, weights.embeddings.pos);

    uint32_t *token_type_ids = (uint32_t *)calloc(num_tokens, sizeof(uint32_t));
    nn_embeddings_forward(&type_out, token_type_ids, num_tokens, weights.embeddings.type);
    free(token_type_ids);

    tensor_binary_op(word_out, pos_out, B_ADD);
    tensor_binary_op(word_out, type_out, B_ADD);

    // layer norm - reuse pos_out as output (it gets overwritten)
    tensor_t ln_gamma = weights.embeddings.ln_gamma;
    tensor_t ln_beta = weights.embeddings.ln_beta;
    int res = nn_layer_norm_forward(&pos_out, word_out, ln_gamma, ln_beta);
    if (res)
    {
        fprintf(stderr, "Failed to layer norm\n");
    }

    tensor_destroy(&word_out);
    tensor_destroy(&type_out);
    tensor_destroy(&position_ids);

    return pos_out;
}

t_status minilm_output_forward(tensor_t *out, const tensor_t hidden_states, const tensor_t input_tensor, struct output_layer_t params)
{
    nn_linear_forward(out, hidden_states, params.weight, params.bias);
    tensor_binary_op(*out, input_tensor, B_ADD);
    nn_layer_norm_forward(out, *out, params.ln_gamma, params.ln_beta);
    return T_OK;
}

t_status minilm_encoder_forward(const tensor_t in, bert_layer_weigts_t weights, tensor_t *out, 
                                const uint32_t *token_ids, size_t num_tokens)
{
    tensor_t q, k, v;
    tensor_t self_out;

    nn_linear_forward(&q, in, weights.query, weights.query_bias);
    nn_linear_forward(&k, in, weights.key, weights.key_bias);
    nn_linear_forward(&v, in, weights.value, weights.value_bias);

    m_try(nn_dot_product_attention_forward(&self_out, q, k, v, 12, token_ids, num_tokens));

    tensor_t tmp;
    minilm_output_forward(&tmp, self_out, in, weights.output);

    // intermediate
    tensor_t intermediate_buffer;
    nn_linear_forward(&intermediate_buffer, tmp, weights.intermediate.weight, weights.intermediate.bias);
    tensor_unary_op(intermediate_buffer, U_GELU, NULL);

    // output
    minilm_output_forward(out, intermediate_buffer, tmp, weights.output_2);

    tensor_destroy(&q);
    tensor_destroy(&k);
    tensor_destroy(&v);
    tensor_destroy(&self_out);
    tensor_destroy(&intermediate_buffer);
    tensor_destroy(&tmp);
    return T_OK;
}

t_status minilm_encode(minilm_t weights, da_u32 ids, tensor_t *out)
{
    tensor_t embedder_out = minilm_embedder_forward(ids, weights);
    for (size_t i = 0; i < 6; i++)
    {
        tensor_t tmp;
        m_try(minilm_encoder_forward(embedder_out, weights.attention[i], &tmp, ids.data, ids.len));
        // Destroy the old embedder_out before reassigning
        tensor_destroy(&embedder_out);
        embedder_out = tmp;
    }
    tensor_t pooled_out;
    nn_mean_pooling(&pooled_out, embedder_out, ids.data, ids.len);
    nn_normalize(&pooled_out);

    tensor_destroy(&embedder_out);

    *out = pooled_out;
    return 0;
}

int minilm_create(minilm_t *m, const char *tbf_path, const char *vocab_txt_path)
{
    m_try(tbf_open(&m->tf, tbf_path));
    minilm_weights_init(m->tf, m);
    m_try(tokenizer_create(&m->tokenizer, vocab_txt_path));
    return 0;
}

void minilm_destroy(minilm_t *m)
{
    tbf_close(m->tf);
    tokenizer_destroy(&m->tokenizer);
}

t_status minilm_embed(minilm_t m, char *str, size_t str_len, tensor_t *out)
{
    s8 str_s8 = s8_from_parts(str, str_len);
    da_u32 ids = {0};
    t_status status = minilm_tokenize(m, str_s8, &ids);
    if (status != T_OK)
    {
        da_u32_free(&ids);
        return status;
    }
    m_try(minilm_encode(m, ids, out));
    da_u32_free(&ids);
    return T_OK;
}
