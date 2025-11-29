// tensor_test.c
#include "tensor.h"
#include <stdio.h>
#include "tbf.h"
#include <stdlib.h>

int main(void)
{
    TbfFile tf;
    // Test runs from build/ directory, so path is relative to there
    t_status status = tbf_open(&tf, "src/test/resources/bert_weights.tbf");
    if (status != T_OK)
    {
        fprintf(stderr, "Failed to open TBF file\n");
        exit(1);
    }

    tensor_t *query = tbf_get_tensor(tf, "encoder.layer.0.attention.self.query.weight");
    if (!query)
    {
        fprintf(stderr, "Failed to get tensor\n");
        tbf_close(tf);
        exit(1);
    }

    printf("query\n");
    tensor_print(*query);

    // new view 12, 384, 32
    tensor_t query_view = tensor_view(3, (uint32_t[]){12, 384, 32}, query->data);
    printf("query_view\n");
    tensor_print(query_view);

    tbf_close(tf);
    return 0;
}
