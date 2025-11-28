#include "minilm.h"
#include "tbf.h"
#include "nn.h"
#include "s8.h"
#include "tokenizer.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

DA(tensor_t)

// --- Distance helper -------------------------------------------------------
// Computes squared L2 distance without modifying the originals.
// We make a temporary copy of `a` once per call, then do in-place ops on it.
// If your tensor API has an explicit copy/clone, use that instead.
static float l2_dist2(const tensor_t a, const tensor_t b)
{
    tensor_t tmp = a; // shallow copy of handle
    tensor_binary_op(tmp, b, B_SUB);
    float pow = 2.0f;
    tensor_unary_op(tmp, U_POW, &pow);
    float d2 = tensor_sum(tmp);
    return d2;
}

// Find index of nearest embedding to `query` using squared L2 distance.
static size_t nearest_index(da_tensor_t vectors, tensor_t query)
{
    size_t best = (size_t)-1;
    float best_d2 = 1e30f;

    for (size_t i = 0; i < vectors.len; i++)
    {
        float d2 = l2_dist2(vectors.data[i], query);
        if (d2 < best_d2)
        {
            best_d2 = d2;
            best = i;
        }
    }
    return best;
}

// --- Embedding helpers ------------------------------------------------------
static void embed_strings(minilm_t *m, const da_s8 *texts, da_tensor_t *out_vecs)
{
    for (size_t i = 0; i < texts->len; i++)
    {
        tensor_t v;
        // Each call fills `v` with the embedding for texts[i]
        minilm_embed(*m, (char *)texts->data[i].data, texts->data[i].len, &v);
        da_tensor_t_append(out_vecs, v);
    }
}

static da_s8 make_choice_list(void)
{
    da_s8 choices = (da_s8){0};
    da_s8_append(&choices, m_s8("paris"));
    da_s8_append(&choices, m_s8("london"));
    da_s8_append(&choices, m_s8("berlin"));
    da_s8_append(&choices, m_s8("madrid"));
    da_s8_append(&choices, m_s8("rome"));
    return choices;
}

// --- Demo -------------------------------------------------------------------
static void run_demo(void)
{
    const char *question = "what's the capital of germany?";

    // 1) Load model + vocab
    minilm_t model;
    minilm_create(&model, "./assets/bert_weights.tbf", "./assets/vocab.txt");

    // 2) Prepare candidate answers and embed them
    da_s8 choices = make_choice_list();
    da_tensor_t choice_vecs = (da_tensor_t){0};
    embed_strings(&model, &choices, &choice_vecs);

    // 3) Embed the query
    tensor_t qvec;
    minilm_embed(model, (char *)question, strlen(question), &qvec);

    // 4) Find nearest neighbor
    size_t idx = nearest_index(choice_vecs, qvec);

    // 5) Print result
    printf("query : %s\n", question);
    printf("answer: %s\n", choices.data[idx].data);

    // 6) Cleanup (if your tensor API exposes frees, call them here)
    // for (size_t i = 0; i < choice_vecs.len; i++) tensor_free(choice_vecs.data[i]);
    // tensor_free(qvec);
    minilm_destroy(&model);
}

int main(void)
{
    run_demo();
    return 0;
}
