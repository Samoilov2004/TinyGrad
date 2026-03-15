#include "tg.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define XOR_BATCH   4
#define XOR_IN      2
#define XOR_HIDDEN  8
#define XOR_OUT     1

#define XOR_STEPS   5000
#define XOR_LR      0.05f

static float frand_uniform(float lo, float hi) {
    float u = (float)rand() / (float)RAND_MAX;
    return lo + (hi - lo) * u;
}

static void fill_zeros(tg_tensor *t) {
    size_t i;
    size_t n;

    if (!t || !t->data) {
        return;
    }

    n = tg_numel(t);
    for (i = 0; i < n; ++i) {
        t->data[i] = 0.0f;
    }
}

static void init_xavier_uniform(tg_tensor *t, int fan_in, int fan_out) {
    size_t i;
    size_t n;
    float limit;

    if (!t || !t->data || fan_in <= 0 || fan_out <= 0) {
        return;
    }

    n = tg_numel(t);
    limit = sqrtf(6.0f / (float)(fan_in + fan_out));

    for (i = 0; i < n; ++i) {
        t->data[i] = frand_uniform(-limit, limit);
    }
}

static tg_tensor *xor_forward(
    tg_arena *arena,
    tg_tensor *x,
    tg_tensor *w1,
    tg_tensor *b1,
    tg_tensor *w2,
    tg_tensor *b2
) {
    tg_tensor *h1_lin;
    tg_tensor *h1;
    tg_tensor *out_lin;
    tg_tensor *logits;

    h1_lin = tg_matmul(arena, x, w1);
    if (!h1_lin) {
        return NULL;
    }

    h1 = tg_add(arena, h1_lin, b1);
    if (!h1) {
        return NULL;
    }

    h1 = tg_relu(arena, h1);
    if (!h1) {
        return NULL;
    }

    out_lin = tg_matmul(arena, h1, w2);
    if (!out_lin) {
        return NULL;
    }

    logits = tg_add(arena, out_lin, b2);
    if (!logits) {
        return NULL;
    }

    return logits;
}

static float run_loss_only(
    tg_arena *arena,
    float *x_data,
    float *y_data,
    tg_tensor *w1,
    tg_tensor *b1,
    tg_tensor *w2,
    tg_tensor *b2
) {
    tg_tensor *x;
    tg_tensor *y;
    tg_tensor *logits;
    tg_tensor *loss;

    x = tg_tensor_from_buffer(arena, x_data, XOR_BATCH, XOR_IN, false);
    y = tg_tensor_from_buffer(arena, y_data, XOR_BATCH, XOR_OUT, false);
    if (!x || !y) {
        return NAN;
    }

    logits = xor_forward(arena, x, w1, b1, w2, b2);
    if (!logits) {
        return NAN;
    }

    loss = tg_bce_with_logits(arena, logits, y);
    if (!loss || !loss->data) {
        return NAN;
    }

    return loss->data[0];
}

static void print_predictions(
    tg_arena *arena,
    float *x_data,
    float *y_data,
    tg_tensor *w1,
    tg_tensor *b1,
    tg_tensor *w2,
    tg_tensor *b2
) {
    int i;
    tg_tensor *x;
    tg_tensor *y;
    tg_tensor *logits;

    x = tg_tensor_from_buffer(arena, x_data, XOR_BATCH, XOR_IN, false);
    y = tg_tensor_from_buffer(arena, y_data, XOR_BATCH, XOR_OUT, false);
    if (!x || !y) {
        printf("prediction forward failed: cannot create input views\n");
        return;
    }

    logits = xor_forward(arena, x, w1, b1, w2, b2);
    if (!logits) {
        printf("prediction forward failed: cannot build graph\n");
        return;
    }

    printf("Predictions:\n");
    for (i = 0; i < XOR_BATCH; ++i) {
        float x0 = x_data[i * XOR_IN + 0];
        float x1 = x_data[i * XOR_IN + 1];
        float target = y_data[i];
        float logit = logits->data[i];
        float prob = 1.0f / (1.0f + expf(-logit));
        int pred = (logit > 0.0f) ? 1 : 0;

        printf("  [%.0f %.0f] -> logit=% .5f prob=%.5f pred=%d target=%.0f\n",
               x0, x1, logit, prob, pred, target);
    }
}

int main(void) {
    /*
    XOR dataset:
      0 xor 0 = 0
      0 xor 1 = 1
      1 xor 0 = 1
      1 xor 1 = 0
    */
    float x_data[XOR_BATCH * XOR_IN] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };

    float y_data[XOR_BATCH * XOR_OUT] = {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };

    tg_arena *arena = NULL;
    tg_param_list params;

    tg_tensor *w1 = NULL;
    tg_tensor *b1 = NULL;
    tg_tensor *w2 = NULL;
    tg_tensor *b2 = NULL;

    int step;
    int ok = 1;

    srand(42);

    arena = tg_arena_create(64 * 1024);
    if (!arena) {
        fprintf(stderr, "failed to create arena\n");
        return 1;
    }

    tg_param_list_init(&params);

    w1 = tg_param_create(XOR_IN, XOR_HIDDEN, true);
    b1 = tg_param_create(1, XOR_HIDDEN, true);
    w2 = tg_param_create(XOR_HIDDEN, XOR_OUT, true);
    b2 = tg_param_create(1, XOR_OUT, true);

    if (!w1 || !b1 || !w2 || !b2) {
        fprintf(stderr, "failed to allocate model parameters\n");
        ok = 0;
        goto cleanup;
    }

    init_xavier_uniform(w1, XOR_IN, XOR_HIDDEN);
    fill_zeros(b1);
    init_xavier_uniform(w2, XOR_HIDDEN, XOR_OUT);
    fill_zeros(b2);

    if (tg_param_list_push(&params, w1) != TG_OK ||
        tg_param_list_push(&params, b1) != TG_OK ||
        tg_param_list_push(&params, w2) != TG_OK ||
        tg_param_list_push(&params, b2) != TG_OK) {
        fprintf(stderr, "failed to build parameter list\n");
        ok = 0;
        goto cleanup;
    }

    printf("Training XOR MLP: 2 -> %d -> 1\n", XOR_HIDDEN);
    printf("optimizer=Adam lr=%.4f steps=%d\n\n", XOR_LR, XOR_STEPS);

    for (step = 1; step <= XOR_STEPS; ++step) {
        tg_tensor *x;
        tg_tensor *y;
        tg_tensor *logits;
        tg_tensor *loss;
        tg_status st;

        tg_param_list_zero_grad(&params);

        x = tg_tensor_from_buffer(arena, x_data, XOR_BATCH, XOR_IN, false);
        y = tg_tensor_from_buffer(arena, y_data, XOR_BATCH, XOR_OUT, false);
        if (!x || !y) {
            fprintf(stderr, "step %d: failed to create input/target views\n", step);
            ok = 0;
            goto cleanup;
        }

        logits = xor_forward(arena, x, w1, b1, w2, b2);
        if (!logits) {
            fprintf(stderr, "step %d: forward failed\n", step);
            ok = 0;
            goto cleanup;
        }

        loss = tg_bce_with_logits(arena, logits, y);
        if (!loss) {
            fprintf(stderr, "step %d: loss creation failed\n", step);
            ok = 0;
            goto cleanup;
        }

        st = tg_backward(loss);
        if (st != TG_OK) {
            fprintf(stderr, "step %d: backward failed with status=%d\n", step, (int)st);
            ok = 0;
            goto cleanup;
        }

        tg_adam_step(
            params.items,
            params.count,
            XOR_LR,
            0.9f,
            0.999f,
            1e-8f,
            step
        );

        if (step == 1 || step % 250 == 0 || step == XOR_STEPS) {
            float eval_loss;

            tg_arena_reset(arena);
            eval_loss = run_loss_only(arena, x_data, y_data, w1, b1, w2, b2);

            if (!isfinite(eval_loss)) {
                fprintf(stderr, "step %d: eval loss is not finite\n", step);
                ok = 0;
                goto cleanup;
            }

            printf("step=%4d  loss=%.6f\n", step, eval_loss);
        }

        tg_arena_reset(arena);
    }

    printf("\nFinal model evaluation:\n");
    print_predictions(arena, x_data, y_data, w1, b1, w2, b2);
    tg_arena_reset(arena);

cleanup:
    tg_param_free(w1);
    tg_param_free(b1);
    tg_param_free(w2);
    tg_param_free(b2);
    tg_param_list_destroy(&params);
    tg_arena_destroy(arena);

    return ok ? 0 : 1;
}
