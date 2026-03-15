#include "tg.h"

#include <math.h>
#include <stddef.h>

static float tg_sigmoid_scalar(float x) {
    /* Численно стабильный sigmoid */
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

static tg_tensor *tg_unary_out_create(
    tg_arena *arena,
    tg_tensor *x,
    tg_backward_fn backward_fn
) {
    if (!arena || !x || !x->data || x->rows <= 0 || x->cols <= 0) {
        return NULL;
    }

    tg_tensor *out = tg_tensor_tmp(arena, x->rows, x->cols, x->requires_grad);
    if (!out) {
        return NULL;
    }

    if (x->requires_grad) {
        tg_op *op = tg_op_create(arena, 1, backward_fn);
        if (!op) {
            return NULL;
        }
        if (tg_op_set_input(op, 0, x) != TG_OK) {
            return NULL;
        }
        out->op = op;
    }

    return out;
}

/*
Backward data source:
- ReLU backward uses OUTPUT data, not input data.
- mask определяется как (out->data[i] > 0), что эквивалентно (x > 0)
  для ReLU = max(0, x).
*/
static void tg_relu_backward(tg_op *op, tg_tensor *out) {
    if (!op || !out || !out->data || !out->grad || op->num_inputs != 1) {
        return;
    }

    tg_tensor *x = op->inputs[0];
    if (!x || !x->requires_grad || !x->grad) {
        return;
    }

    size_t n = tg_numel(x);
    if (n == 0 || n != tg_numel(out)) {
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        if (out->data[i] > 0.0f) {
            x->grad[i] += out->grad[i];
        }
    }
}

/*
Backward data source:
- Sigmoid backward uses OUTPUT data only:
    s = out->data[i]
    dx += grad_out * s * (1 - s)
*/
static void tg_sigmoid_backward(tg_op *op, tg_tensor *out) {
    if (!op || !out || !out->data || !out->grad || op->num_inputs != 1) {
        return;
    }

    tg_tensor *x = op->inputs[0];
    if (!x || !x->requires_grad || !x->grad) {
        return;
    }

    size_t n = tg_numel(x);
    if (n == 0 || n != tg_numel(out)) {
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        float s = out->data[i];
        x->grad[i] += out->grad[i] * s * (1.0f - s);
    }
}

/*
Backward data source:
- Tanh backward uses OUTPUT data only:
    t = out->data[i]
    dx += grad_out * (1 - t^2)
*/
static void tg_tanh_backward(tg_op *op, tg_tensor *out) {
    if (!op || !out || !out->data || !out->grad || op->num_inputs != 1) {
        return;
    }

    tg_tensor *x = op->inputs[0];
    if (!x || !x->requires_grad || !x->grad) {
        return;
    }

    size_t n = tg_numel(x);
    if (n == 0 || n != tg_numel(out)) {
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        float t = out->data[i];
        x->grad[i] += out->grad[i] * (1.0f - t * t);
    }
}

tg_tensor *tg_relu(tg_arena *arena, tg_tensor *x) {
    tg_tensor *out = tg_unary_out_create(arena, x, tg_relu_backward);
    if (!out) {
        return NULL;
    }

    size_t n = tg_numel(x);
    for (size_t i = 0; i < n; ++i) {
        float v = x->data[i];
        out->data[i] = (v > 0.0f) ? v : 0.0f;
    }

    return out;
}

tg_tensor *tg_sigmoid(tg_arena *arena, tg_tensor *x) {
    tg_tensor *out = tg_unary_out_create(arena, x, tg_sigmoid_backward);
    if (!out) {
        return NULL;
    }

    size_t n = tg_numel(x);
    for (size_t i = 0; i < n; ++i) {
        out->data[i] = tg_sigmoid_scalar(x->data[i]);
    }

    return out;
}

tg_tensor *tg_tanh(tg_arena *arena, tg_tensor *x) {
    tg_tensor *out = tg_unary_out_create(arena, x, tg_tanh_backward);
    if (!out) {
        return NULL;
    }

    size_t n = tg_numel(x);
    for (size_t i = 0; i < n; ++i) {
        out->data[i] = tanhf(x->data[i]);
    }

    return out;
}
