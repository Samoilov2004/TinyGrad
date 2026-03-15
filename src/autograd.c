#include "tg.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef struct tg_tensor_vec {
    tg_tensor **data;
    size_t len;
    size_t cap;
} tg_tensor_vec;

typedef struct tg_frame {
    tg_tensor *tensor;
    int next_input_idx;
} tg_frame;

typedef struct tg_frame_vec {
    tg_frame *data;
    size_t len;
    size_t cap;
} tg_frame_vec;

typedef struct tg_visit_entry {
    tg_tensor *tensor;
    unsigned char state; /* 1 = entered, 2 = done */
} tg_visit_entry;

typedef struct tg_visit_vec {
    tg_visit_entry *data;
    size_t len;
    size_t cap;
} tg_visit_vec;

enum {
    TG_VISIT_ENTERED = 1,
    TG_VISIT_DONE = 2
};

static int tg_size_mul_overflow(size_t a, size_t b) {
    return (a != 0 && b > ((size_t)-1) / a);
}

static tg_status tg_tensor_vec_reserve(tg_tensor_vec *vec, size_t min_cap) {
    size_t new_cap;
    tg_tensor **new_data;

    if (vec == NULL) {
        return TG_ERR_INVALID_ARGUMENT;
    }
    if (vec->cap >= min_cap) {
        return TG_OK;
    }

    new_cap = (vec->cap == 0) ? 16u : vec->cap;
    while (new_cap < min_cap) {
        if (new_cap > ((size_t)-1) / 2u) {
            new_cap = min_cap;
            break;
        }
        new_cap *= 2u;
    }

    if (tg_size_mul_overflow(new_cap, sizeof(*new_data))) {
        return TG_ERR_OOM;
    }

    new_data = (tg_tensor **)realloc(vec->data, new_cap * sizeof(*new_data));
    if (new_data == NULL) {
        return TG_ERR_OOM;
    }

    vec->data = new_data;
    vec->cap = new_cap;
    return TG_OK;
}

static tg_status tg_frame_vec_reserve(tg_frame_vec *vec, size_t min_cap) {
    size_t new_cap;
    tg_frame *new_data;

    if (vec == NULL) {
        return TG_ERR_INVALID_ARGUMENT;
    }
    if (vec->cap >= min_cap) {
        return TG_OK;
    }

    new_cap = (vec->cap == 0) ? 16u : vec->cap;
    while (new_cap < min_cap) {
        if (new_cap > ((size_t)-1) / 2u) {
            new_cap = min_cap;
            break;
        }
        new_cap *= 2u;
    }

    if (tg_size_mul_overflow(new_cap, sizeof(*new_data))) {
        return TG_ERR_OOM;
    }

    new_data = (tg_frame *)realloc(vec->data, new_cap * sizeof(*new_data));
    if (new_data == NULL) {
        return TG_ERR_OOM;
    }

    vec->data = new_data;
    vec->cap = new_cap;
    return TG_OK;
}

static tg_status tg_visit_vec_reserve(tg_visit_vec *vec, size_t min_cap) {
    size_t new_cap;
    tg_visit_entry *new_data;

    if (vec == NULL) {
        return TG_ERR_INVALID_ARGUMENT;
    }
    if (vec->cap >= min_cap) {
        return TG_OK;
    }

    new_cap = (vec->cap == 0) ? 16u : vec->cap;
    while (new_cap < min_cap) {
        if (new_cap > ((size_t)-1) / 2u) {
            new_cap = min_cap;
            break;
        }
        new_cap *= 2u;
    }

    if (tg_size_mul_overflow(new_cap, sizeof(*new_data))) {
        return TG_ERR_OOM;
    }

    new_data = (tg_visit_entry *)realloc(vec->data, new_cap * sizeof(*new_data));
    if (new_data == NULL) {
        return TG_ERR_OOM;
    }

    vec->data = new_data;
    vec->cap = new_cap;
    return TG_OK;
}

static void tg_tensor_vec_free(tg_tensor_vec *vec) {
    if (vec != NULL) {
        free(vec->data);
        vec->data = NULL;
        vec->len = 0;
        vec->cap = 0;
    }
}

static void tg_frame_vec_free(tg_frame_vec *vec) {
    if (vec != NULL) {
        free(vec->data);
        vec->data = NULL;
        vec->len = 0;
        vec->cap = 0;
    }
}

static void tg_visit_vec_free(tg_visit_vec *vec) {
    if (vec != NULL) {
        free(vec->data);
        vec->data = NULL;
        vec->len = 0;
        vec->cap = 0;
    }
}

static tg_status tg_tensor_vec_push(tg_tensor_vec *vec, tg_tensor *tensor) {
    tg_status st;

    if (vec == NULL) {
        return TG_ERR_INVALID_ARGUMENT;
    }

    st = tg_tensor_vec_reserve(vec, vec->len + 1u);
    if (st != TG_OK) {
        return st;
    }

    vec->data[vec->len++] = tensor;
    return TG_OK;
}

static tg_status tg_frame_vec_push(tg_frame_vec *vec, tg_frame frame) {
    tg_status st;

    if (vec == NULL) {
        return TG_ERR_INVALID_ARGUMENT;
    }

    st = tg_frame_vec_reserve(vec, vec->len + 1u);
    if (st != TG_OK) {
        return st;
    }

    vec->data[vec->len++] = frame;
    return TG_OK;
}

static ptrdiff_t tg_visit_find(const tg_visit_vec *visited, const tg_tensor *tensor) {
    size_t i;

    if (visited == NULL || tensor == NULL) {
        return -1;
    }

    for (i = 0; i < visited->len; ++i) {
        if (visited->data[i].tensor == tensor) {
            return (ptrdiff_t)i;
        }
    }

    return -1;
}

static unsigned char tg_visit_state(const tg_visit_vec *visited, const tg_tensor *tensor) {
    ptrdiff_t idx = tg_visit_find(visited, tensor);
    if (idx < 0) {
        return 0;
    }
    return visited->data[(size_t)idx].state;
}

static tg_status tg_visit_set(tg_visit_vec *visited, tg_tensor *tensor, unsigned char state) {
    ptrdiff_t idx;
    tg_status st;

    if (visited == NULL || tensor == NULL) {
        return TG_ERR_INVALID_ARGUMENT;
    }

    idx = tg_visit_find(visited, tensor);
    if (idx >= 0) {
        visited->data[(size_t)idx].state = state;
        return TG_OK;
    }

    st = tg_visit_vec_reserve(visited, visited->len + 1u);
    if (st != TG_OK) {
        return st;
    }

    visited->data[visited->len].tensor = tensor;
    visited->data[visited->len].state = state;
    visited->len += 1u;
    return TG_OK;
}

/*
Построение topo-order без рекурсии.

Результат:
- topo = [leaf ..., loss]
*/
static tg_status tg_build_topo(tg_tensor *root, tg_tensor_vec *topo) {
    tg_frame_vec stack = {0};
    tg_visit_vec visited = {0};
    tg_status st = TG_OK;

    if (root == NULL || topo == NULL) {
        return TG_ERR_INVALID_ARGUMENT;
    }

    st = tg_visit_set(&visited, root, TG_VISIT_ENTERED);
    if (st != TG_OK) {
        goto cleanup;
    }

    st = tg_frame_vec_push(&stack, (tg_frame){ root, 0 });
    if (st != TG_OK) {
        goto cleanup;
    }

    while (stack.len > 0u) {
        tg_frame *frame = &stack.data[stack.len - 1u];
        tg_tensor *t = frame->tensor;
        tg_op *op;

        if (t == NULL) {
            st = TG_ERR_INVALID_ARGUMENT;
            goto cleanup;
        }

        op = t->op;
        if (op == NULL) {
            st = tg_visit_set(&visited, t, TG_VISIT_DONE);
            if (st != TG_OK) {
                goto cleanup;
            }

            st = tg_tensor_vec_push(topo, t);
            if (st != TG_OK) {
                goto cleanup;
            }

            stack.len -= 1u;
            continue;
        }

        if (op->num_inputs < 0 || op->num_inputs > TG_OP_MAX_INPUTS) {
            st = TG_ERR_INVALID_ARGUMENT;
            goto cleanup;
        }

        if (frame->next_input_idx < op->num_inputs) {
            tg_tensor *in = op->inputs[frame->next_input_idx++];
            unsigned char state;

            if (in == NULL) {
                continue;
            }

            state = tg_visit_state(&visited, in);
            if (state == 0) {
                st = tg_visit_set(&visited, in, TG_VISIT_ENTERED);
                if (st != TG_OK) {
                    goto cleanup;
                }

                st = tg_frame_vec_push(&stack, (tg_frame){ in, 0 });
                if (st != TG_OK) {
                    goto cleanup;
                }
            } else if (state == TG_VISIT_ENTERED) {
                /* Циклы не поддерживаются: граф должен быть DAG. */
                st = TG_ERR_INVALID_ARGUMENT;
                goto cleanup;
            }

            continue;
        }

        st = tg_visit_set(&visited, t, TG_VISIT_DONE);
        if (st != TG_OK) {
            goto cleanup;
        }

        st = tg_tensor_vec_push(topo, t);
        if (st != TG_OK) {
            goto cleanup;
        }

        stack.len -= 1u;
    }

cleanup:
    tg_frame_vec_free(&stack);
    tg_visit_vec_free(&visited);
    return st;
}

tg_op *tg_op_create(tg_arena *arena, int num_inputs, tg_backward_fn backward) {
    tg_op *op;

    if (arena == NULL) {
        return NULL;
    }
    if (num_inputs < 0 || num_inputs > TG_OP_MAX_INPUTS) {
        return NULL;
    }
    if (backward == NULL) {
        return NULL;
    }

    op = (tg_op *)tg_arena_alloc(arena, sizeof(*op), _Alignof(tg_op));
    if (op == NULL) {
        return NULL;
    }

    memset(op, 0, sizeof(*op));
    op->num_inputs = num_inputs;
    op->backward = backward;
    return op;
}

tg_status tg_op_set_input(tg_op *op, int index, tg_tensor *input) {
    if (op == NULL) {
        return TG_ERR_INVALID_ARGUMENT;
    }
    if (index < 0 || index >= op->num_inputs || index >= TG_OP_MAX_INPUTS) {
        return TG_ERR_INVALID_ARGUMENT;
    }

    op->inputs[index] = input;
    return TG_OK;
}

tg_status tg_backward(tg_tensor *loss) {
    tg_tensor_vec topo = {0};
    tg_status st;
    size_t i;

    if (loss == NULL) {
        return TG_ERR_INVALID_ARGUMENT;
    }

    /*
    MVP:
    - поддерживаем только scalar loss
    - если потом понадобится, можно расширить до seed = 1 для всех элементов
    */
    if (tg_numel(loss) != 1u) {
        return TG_ERR_INVALID_ARGUMENT;
    }

    if (!loss->requires_grad) {
        return TG_ERR_INVALID_ARGUMENT;
    }

    if (loss->grad == NULL) {
        return TG_ERR_INVALID_ARGUMENT;
    }

    st = tg_build_topo(loss, &topo);
    if (st != TG_OK) {
        tg_tensor_vec_free(&topo);
        return st;
    }

    /* Seed dL/dL = 1 */
    loss->grad[0] = 1.0f;

    /*
    topo: [leaf ..., loss]
    backward order: [loss ... leaf]
    */
    for (i = topo.len; i > 0u; --i) {
        tg_tensor *out = topo.data[i - 1u];
        tg_op *op;

        if (out == NULL) {
            st = TG_ERR_INVALID_ARGUMENT;
            break;
        }

        op = out->op;
        if (op == NULL) {
            continue;
        }

        if (op->backward == NULL) {
            st = TG_ERR_INVALID_ARGUMENT;
            break;
        }

        /*
        Если у out нет grad buffer, backward callback не сможет прочитать dL/dout.
        Для корректного графа такого обычно не должно быть.
        */
        if (out->grad == NULL) {
            continue;
        }

        op->backward(op, out);
    }

    tg_tensor_vec_free(&topo);
    return st;
}
