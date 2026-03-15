#include "tg.h"

#include <stddef.h>

static bool tg_tensor_valid_data(const tg_tensor *t) {
    return t != NULL &&
           t->data != NULL &&
           t->rows > 0 &&
           t->cols > 0;
}

static bool tg_same_shape(const tg_tensor *a, const tg_tensor *b) {
    return a != NULL &&
           b != NULL &&
           a->rows == b->rows &&
           a->cols == b->cols;
}

static bool tg_bias_shape(const tg_tensor *a, const tg_tensor *b) {
    return a != NULL &&
           b != NULL &&
           a->rows > 0 &&
           a->cols > 0 &&
           b->rows == 1 &&
           b->cols == a->cols;
}

static tg_tensor *tg_make_binary_out(
    tg_arena *arena,
    int rows,
    int cols,
    tg_tensor *a,
    tg_tensor *b,
    tg_backward_fn backward
) {
    tg_tensor *out;
    tg_op *op;
    tg_status st;
    bool requires_grad;

    if (arena == NULL || a == NULL || b == NULL) {
        return NULL;
    }

    requires_grad = a->requires_grad || b->requires_grad;
    out = tg_tensor_tmp(arena, rows, cols, requires_grad);
    if (out == NULL) {
        return NULL;
    }

    out->op = NULL;

    if (!requires_grad) {
        return out;
    }

    op = tg_op_create(arena, 2, backward);
    if (op == NULL) {
        return NULL;
    }

    st = tg_op_set_input(op, 0, a);
    if (st != TG_OK) {
        return NULL;
    }

    st = tg_op_set_input(op, 1, b);
    if (st != TG_OK) {
        return NULL;
    }

    out->op = op;
    return out;
}

static tg_tensor *tg_make_unary_out(
    tg_arena *arena,
    int rows,
    int cols,
    tg_tensor *x,
    tg_backward_fn backward
) {
    tg_tensor *out;
    tg_op *op;
    tg_status st;

    if (arena == NULL || x == NULL) {
        return NULL;
    }

    out = tg_tensor_tmp(arena, rows, cols, x->requires_grad);
    if (out == NULL) {
        return NULL;
    }

    out->op = NULL;

    if (!x->requires_grad) {
        return out;
    }

    op = tg_op_create(arena, 1, backward);
    if (op == NULL) {
        return NULL;
    }

    st = tg_op_set_input(op, 0, x);
    if (st != TG_OK) {
        return NULL;
    }

    out->op = op;
    return out;
}

static void tg_addsub_backward_common(tg_op *op, tg_tensor *out, float sign_b) {
    tg_tensor *a;
    tg_tensor *b;
    int r, c;

    if (op == NULL || out == NULL || out->grad == NULL) {
        return;
    }

    a = op->inputs[0];
    b = op->inputs[1];

    if (a == NULL || b == NULL) {
        return;
    }

    if (tg_same_shape(a, b)) {
        size_t n;
        size_t i;

        if (a->rows != out->rows || a->cols != out->cols) {
            return;
        }

        n = tg_numel(out);

        if (a->requires_grad && a->grad != NULL) {
            for (i = 0; i < n; ++i) {
                a->grad[i] += out->grad[i];
            }
        }

        if (b->requires_grad && b->grad != NULL) {
            for (i = 0; i < n; ++i) {
                b->grad[i] += sign_b * out->grad[i];
            }
        }

        return;
    }

    if (tg_bias_shape(a, b)) {
        if (out->rows != a->rows || out->cols != a->cols) {
            return;
        }

        if (a->requires_grad && a->grad != NULL) {
            for (r = 0; r < out->rows; ++r) {
                for (c = 0; c < out->cols; ++c) {
                    size_t idx = (size_t)r * (size_t)out->cols + (size_t)c;
                    a->grad[idx] += out->grad[idx];
                }
            }
        }

        if (b->requires_grad && b->grad != NULL) {
            for (c = 0; c < out->cols; ++c) {
                float acc = 0.0f;
                for (r = 0; r < out->rows; ++r) {
                    size_t idx = (size_t)r * (size_t)out->cols + (size_t)c;
                    acc += out->grad[idx];
                }
                b->grad[c] += sign_b * acc;
            }
        }
    }
}

static void tg_add_backward(tg_op *op, tg_tensor *out) {
    tg_addsub_backward_common(op, out, 1.0f);
}

static void tg_sub_backward(tg_op *op, tg_tensor *out) {
    tg_addsub_backward_common(op, out, -1.0f);
}

static void tg_mul_backward(tg_op *op, tg_tensor *out) {
    tg_tensor *a;
    tg_tensor *b;
    size_t n;
    size_t i;

    if (op == NULL || out == NULL || out->grad == NULL) {
        return;
    }

    a = op->inputs[0];
    b = op->inputs[1];

    if (a == NULL || b == NULL) {
        return;
    }

    if (!tg_same_shape(a, b)) {
        return;
    }

    if (a->rows != out->rows || a->cols != out->cols) {
        return;
    }

    n = tg_numel(out);

    if (a->requires_grad && a->grad != NULL) {
        for (i = 0; i < n; ++i) {
            a->grad[i] += out->grad[i] * b->data[i];
        }
    }

    if (b->requires_grad && b->grad != NULL) {
        for (i = 0; i < n; ++i) {
            b->grad[i] += out->grad[i] * a->data[i];
        }
    }
}

static void tg_matmul_backward(tg_op *op, tg_tensor *out) {
    tg_tensor *a;
    tg_tensor *b;
    int batch, d, h;
    int i, k, j;

    if (op == NULL || out == NULL || out->grad == NULL) {
        return;
    }

    a = op->inputs[0];
    b = op->inputs[1];

    if (a == NULL || b == NULL) {
        return;
    }

    if (a->cols != b->rows) {
        return;
    }

    batch = a->rows;
    d = a->cols;
    h = b->cols;

    if (out->rows != batch || out->cols != h) {
        return;
    }

    /*
    dA = dOut * B^T
    A:    [batch x d]
    B:    [d x h]
    dOut: [batch x h]
    */
    if (a->requires_grad && a->grad != NULL) {
        for (i = 0; i < batch; ++i) {
            for (k = 0; k < d; ++k) {
                float acc = 0.0f;
                for (j = 0; j < h; ++j) {
                    size_t out_idx = (size_t)i * (size_t)h + (size_t)j;
                    size_t b_idx = (size_t)k * (size_t)h + (size_t)j;
                    acc += out->grad[out_idx] * b->data[b_idx];
                }
                a->grad[(size_t)i * (size_t)d + (size_t)k] += acc;
            }
        }
    }

    /*
    dB = A^T * dOut
    */
    if (b->requires_grad && b->grad != NULL) {
        for (k = 0; k < d; ++k) {
            for (j = 0; j < h; ++j) {
                float acc = 0.0f;
                for (i = 0; i < batch; ++i) {
                    size_t a_idx = (size_t)i * (size_t)d + (size_t)k;
                    size_t out_idx = (size_t)i * (size_t)h + (size_t)j;
                    acc += a->data[a_idx] * out->grad[out_idx];
                }
                b->grad[(size_t)k * (size_t)h + (size_t)j] += acc;
            }
        }
    }
}

static void tg_sum_backward(tg_op *op, tg_tensor *out) {
    tg_tensor *x;
    size_t n;
    size_t i;
    float g;

    if (op == NULL || out == NULL || out->grad == NULL) {
        return;
    }

    x = op->inputs[0];
    if (x == NULL || !x->requires_grad || x->grad == NULL) {
        return;
    }

    n = tg_numel(x);
    g = out->grad[0];

    for (i = 0; i < n; ++i) {
        x->grad[i] += g;
    }
}

static void tg_mean_backward(tg_op *op, tg_tensor *out) {
    tg_tensor *x;
    size_t n;
    size_t i;
    float g;

    if (op == NULL || out == NULL || out->grad == NULL) {
        return;
    }

    x = op->inputs[0];
    if (x == NULL || !x->requires_grad || x->grad == NULL) {
        return;
    }

    n = tg_numel(x);
    if (n == 0u) {
        return;
    }

    g = out->grad[0] / (float)n;

    for (i = 0; i < n; ++i) {
        x->grad[i] += g;
    }
}

tg_tensor *tg_add(tg_arena *arena, tg_tensor *a, tg_tensor *b) {
    tg_tensor *out;
    int r, c;

    if (arena == NULL || !tg_tensor_valid_data(a) || !tg_tensor_valid_data(b)) {
        return NULL;
    }

    if (!tg_same_shape(a, b) && !tg_bias_shape(a, b)) {
        return NULL;
    }

    out = tg_make_binary_out(arena, a->rows, a->cols, a, b, tg_add_backward);
    if (out == NULL) {
        return NULL;
    }

    if (tg_same_shape(a, b)) {
        size_t n = tg_numel(out);
        size_t i;
        for (i = 0; i < n; ++i) {
            out->data[i] = a->data[i] + b->data[i];
        }
    } else {
        for (r = 0; r < a->rows; ++r) {
            for (c = 0; c < a->cols; ++c) {
                size_t idx = (size_t)r * (size_t)a->cols + (size_t)c;
                out->data[idx] = a->data[idx] + b->data[c];
            }
        }
    }

    return out;
}

tg_tensor *tg_sub(tg_arena *arena, tg_tensor *a, tg_tensor *b) {
    tg_tensor *out;
    int r, c;

    if (arena == NULL || !tg_tensor_valid_data(a) || !tg_tensor_valid_data(b)) {
        return NULL;
    }

    if (!tg_same_shape(a, b) && !tg_bias_shape(a, b)) {
        return NULL;
    }

    out = tg_make_binary_out(arena, a->rows, a->cols, a, b, tg_sub_backward);
    if (out == NULL) {
        return NULL;
    }

    if (tg_same_shape(a, b)) {
        size_t n = tg_numel(out);
        size_t i;
        for (i = 0; i < n; ++i) {
            out->data[i] = a->data[i] - b->data[i];
        }
    } else {
        for (r = 0; r < a->rows; ++r) {
            for (c = 0; c < a->cols; ++c) {
                size_t idx = (size_t)r * (size_t)a->cols + (size_t)c;
                out->data[idx] = a->data[idx] - b->data[c];
            }
        }
    }

    return out;
}

tg_tensor *tg_mul(tg_arena *arena, tg_tensor *a, tg_tensor *b) {
    tg_tensor *out;
    size_t n;
    size_t i;

    if (arena == NULL || !tg_tensor_valid_data(a) || !tg_tensor_valid_data(b)) {
        return NULL;
    }

    if (!tg_same_shape(a, b)) {
        return NULL;
    }

    out = tg_make_binary_out(arena, a->rows, a->cols, a, b, tg_mul_backward);
    if (out == NULL) {
        return NULL;
    }

    n = tg_numel(out);
    for (i = 0; i < n; ++i) {
        out->data[i] = a->data[i] * b->data[i];
    }

    return out;
}

tg_tensor *tg_matmul(tg_arena *arena, tg_tensor *a, tg_tensor *b) {
    tg_tensor *out;
    int batch, d, h;
    int i, k, j;

    if (arena == NULL || !tg_tensor_valid_data(a) || !tg_tensor_valid_data(b)) {
        return NULL;
    }

    if (a->cols != b->rows) {
        return NULL;
    }

    batch = a->rows;
    d = a->cols;
    h = b->cols;

    out = tg_make_binary_out(arena, batch, h, a, b, tg_matmul_backward);
    if (out == NULL) {
        return NULL;
    }

    for (i = 0; i < batch; ++i) {
        for (j = 0; j < h; ++j) {
            float acc = 0.0f;
            for (k = 0; k < d; ++k) {
                size_t a_idx = (size_t)i * (size_t)d + (size_t)k;
                size_t b_idx = (size_t)k * (size_t)h + (size_t)j;
                acc += a->data[a_idx] * b->data[b_idx];
            }
            out->data[(size_t)i * (size_t)h + (size_t)j] = acc;
        }
    }

    return out;
}

tg_tensor *tg_sum(tg_arena *arena, tg_tensor *x) {
    tg_tensor *out;
    size_t n;
    size_t i;
    float acc = 0.0f;

    if (arena == NULL || !tg_tensor_valid_data(x)) {
        return NULL;
    }

    out = tg_make_unary_out(arena, 1, 1, x, tg_sum_backward);
    if (out == NULL) {
        return NULL;
    }

    n = tg_numel(x);
    for (i = 0; i < n; ++i) {
        acc += x->data[i];
    }

    out->data[0] = acc;
    return out;
}

tg_tensor *tg_mean(tg_arena *arena, tg_tensor *x) {
    tg_tensor *out;
    size_t n;
    size_t i;
    float acc = 0.0f;

    if (arena == NULL || !tg_tensor_valid_data(x)) {
        return NULL;
    }

    n = tg_numel(x);
    if (n == 0u) {
        return NULL;
    }

    out = tg_make_unary_out(arena, 1, 1, x, tg_mean_backward);
    if (out == NULL) {
        return NULL;
    }

    for (i = 0; i < n; ++i) {
        acc += x->data[i];
    }

    out->data[0] = acc / (float)n;
    return out;
}
