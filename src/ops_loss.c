#include "tg.h"

#include <math.h>
#include <stddef.h>

static float tg_sigmoid_scalar(float x) {
    /* Численно стабильный sigmoid для BCE backward */
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

static bool tg_same_shape(const tg_tensor *a, const tg_tensor *b) {
    return a && b &&
           a->rows > 0 && a->cols > 0 &&
           b->rows > 0 && b->cols > 0 &&
           a->rows == b->rows &&
           a->cols == b->cols;
}

static tg_tensor *tg_scalar_loss_out_create(
    tg_arena *arena,
    bool requires_grad,
    tg_backward_fn backward_fn,
    tg_tensor *in0,
    tg_tensor *in1
) {
    if (!arena || !in0 || !in1) {
        return NULL;
    }

    tg_tensor *out = tg_tensor_tmp(arena, 1, 1, requires_grad);
    if (!out) {
        return NULL;
    }

    if (requires_grad) {
        tg_op *op = tg_op_create(arena, 2, backward_fn);
        if (!op) {
            return NULL;
        }
        if (tg_op_set_input(op, 0, in0) != TG_OK) {
            return NULL;
        }
        if (tg_op_set_input(op, 1, in1) != TG_OK) {
            return NULL;
        }
        out->op = op;
    }

    return out;
}

/*
Backward data source:
- uses INPUT tensors pred and target
- gradient is accumulated only into pred
- target is treated as constant label tensor
*/
static void tg_mse_backward(tg_op *op, tg_tensor *out) {
    if (!op || !out || !out->grad || op->num_inputs != 2) {
        return;
    }

    tg_tensor *pred = op->inputs[0];
    tg_tensor *target = op->inputs[1];
    if (!pred || !target || !pred->data || !target->data) {
        return;
    }
    if (!pred->requires_grad || !pred->grad) {
        return;
    }

    size_t n = tg_numel(pred);
    if (n == 0 || n != tg_numel(target)) {
        return;
    }

    float scale = (2.0f / (float)n) * out->grad[0];
    for (size_t i = 0; i < n; ++i) {
        pred->grad[i] += scale * (pred->data[i] - target->data[i]);
    }
}

/*
Backward data source:
- uses INPUT logits and target
- derivative: sigmoid(logits) - target
- gradient is accumulated only into logits
- target is treated as constant label tensor
*/
static void tg_bce_with_logits_backward(tg_op *op, tg_tensor *out) {
    if (!op || !out || !out->grad || op->num_inputs != 2) {
        return;
    }

    tg_tensor *logits = op->inputs[0];
    tg_tensor *target = op->inputs[1];
    if (!logits || !target || !logits->data || !target->data) {
        return;
    }
    if (!logits->requires_grad || !logits->grad) {
        return;
    }

    size_t n = tg_numel(logits);
    if (n == 0 || n != tg_numel(target)) {
        return;
    }

    float scale = out->grad[0] / (float)n;
    for (size_t i = 0; i < n; ++i) {
        float s = tg_sigmoid_scalar(logits->data[i]);
        logits->grad[i] += scale * (s - target->data[i]);
    }
}

/*
Backward data source:
- uses INPUT logits and target_onehot
- stable softmax is recomputed row-wise from logits during backward
- no extra cache is required
- gradient is accumulated only into logits
- expected shapes:
    logits:        [B x C]
    target_onehot: [B x C]
- dlogits += (softmax - target) / B * grad_out
*/
static void tg_softmax_cross_entropy_backward(tg_op *op, tg_tensor *out) {
    if (!op || !out || !out->grad || op->num_inputs != 2) {
        return;
    }

    tg_tensor *logits = op->inputs[0];
    tg_tensor *target = op->inputs[1];
    if (!logits || !target || !logits->data || !target->data) {
        return;
    }
    if (!logits->requires_grad || !logits->grad) {
        return;
    }
    if (logits->rows <= 0 || logits->cols <= 0) {
        return;
    }
    if (!tg_same_shape(logits, target)) {
        return;
    }

    int B = logits->rows;
    int C = logits->cols;
    float scale = out->grad[0] / (float)B;

    for (int r = 0; r < B; ++r) {
        int row_off = r * C;

        float row_max = logits->data[row_off];
        for (int c = 1; c < C; ++c) {
            float v = logits->data[row_off + c];
            if (v > row_max) {
                row_max = v;
            }
        }

        double sumexp = 0.0;
        for (int c = 0; c < C; ++c) {
            sumexp += (double)expf(logits->data[row_off + c] - row_max);
        }

        for (int c = 0; c < C; ++c) {
            float p = (float)(expf(logits->data[row_off + c] - row_max) / (float)sumexp);
            logits->grad[row_off + c] += scale * (p - target->data[row_off + c]);
        }
    }
}

tg_tensor *tg_mse(tg_arena *arena, tg_tensor *pred, tg_tensor *target) {
    if (!arena || !tg_same_shape(pred, target) || !pred->data || !target->data) {
        return NULL;
    }

    size_t n = tg_numel(pred);
    if (n == 0) {
        return NULL;
    }

    /*
    Для loss target считаем константой:
    out.requires_grad зависит только от pred.
    */
    tg_tensor *out = tg_scalar_loss_out_create(
        arena,
        pred->requires_grad,
        tg_mse_backward,
        pred,
        target
    );
    if (!out) {
        return NULL;
    }

    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double)pred->data[i] - (double)target->data[i];
        acc += d * d;
    }

    out->data[0] = (float)(acc / (double)n);
    return out;
}

tg_tensor *tg_bce_with_logits(tg_arena *arena, tg_tensor *logits, tg_tensor *target) {
    if (!arena || !tg_same_shape(logits, target) || !logits->data || !target->data) {
        return NULL;
    }

    size_t n = tg_numel(logits);
    if (n == 0) {
        return NULL;
    }

    /*
    Ожидаемые формы:
    - logits: [R x C]
    - target: [R x C]
    Возвращает mean BCE по всем N = R*C элементам.
    Target считается константой.
    */
    tg_tensor *out = tg_scalar_loss_out_create(
        arena,
        logits->requires_grad,
        tg_bce_with_logits_backward,
        logits,
        target
    );
    if (!out) {
        return NULL;
    }

    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float x = logits->data[i];
        float y = target->data[i];

        /*
        stable BCE-with-logits:
            max(x, 0) - x*y + log(1 + exp(-|x|))
        */
        float loss_i = fmaxf(x, 0.0f) - x * y + log1pf(expf(-fabsf(x)));
        acc += (double)loss_i;
    }

    out->data[0] = (float)(acc / (double)n);
    return out;
}

tg_tensor *tg_softmax_cross_entropy(
    tg_arena *arena,
    tg_tensor *logits,
    tg_tensor *target_onehot
) {
    if (!arena || !tg_same_shape(logits, target_onehot) ||
        !logits->data || !target_onehot->data) {
        return NULL;
    }
    if (logits->rows <= 0 || logits->cols <= 0) {
        return NULL;
    }

    /*
    Ожидаемые формы:
    - logits:        [B x C]
    - target_onehot: [B x C]

    Возвращает scalar [1 x 1]:
    средний cross-entropy loss по batch B.

    Реализация fused/stable:
    - отдельный softmax tensor не создаётся
    - используется row-wise log-sum-exp trick
    */
    tg_tensor *out = tg_scalar_loss_out_create(
        arena,
        logits->requires_grad,
        tg_softmax_cross_entropy_backward,
        logits,
        target_onehot
    );
    if (!out) {
        return NULL;
    }

    int B = logits->rows;
    int C = logits->cols;

    double total = 0.0;

    for (int r = 0; r < B; ++r) {
        int row_off = r * C;

        float row_max = logits->data[row_off];
        for (int c = 1; c < C; ++c) {
            float v = logits->data[row_off + c];
            if (v > row_max) {
                row_max = v;
            }
        }

        double sumexp = 0.0;
        for (int c = 0; c < C; ++c) {
            sumexp += (double)expf(logits->data[row_off + c] - row_max);
        }

        double logsumexp = (double)row_max + log(sumexp);

        double row_loss = 0.0;
        for (int c = 0; c < C; ++c) {
            /*
            Для one-hot:
                CE = -sum target * log_softmax
                   = sum target * (logsumexp - logits)
            Также работает для soft targets, если строка target суммируется к 1.
            */
            row_loss += (double)target_onehot->data[row_off + c] *
                        (logsumexp - (double)logits->data[row_off + c]);
        }

        total += row_loss;
    }

    out->data[0] = (float)(total / (double)B);
    return out;
}
