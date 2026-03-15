#include "tg.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static bool tg_param_is_heap_backed(const tg_tensor *p) {
    if (!p) {
        return false;
    }
    if (!p->data) {
        return false;
    }
    if (p->rows <= 0 || p->cols <= 0) {
        return false;
    }

    /*
    Основная защита от попадания arena tensors в список параметров:
    heap-параметры обычно владеют собой и data.
    */
    if (!p->owns_self || !p->owns_data) {
        return false;
    }

    return true;
}

static bool tg_param_is_updatable(const tg_tensor *p) {
    if (!tg_param_is_heap_backed(p)) {
        return false;
    }
    if (!p->requires_grad) {
        return false;
    }
    if (!p->grad) {
        return false;
    }
    return true;
}

static float *tg_alloc_state_zeroed(size_t n) {
    if (n == 0) {
        return NULL;
    }
    return (float *)calloc(n, sizeof(float));
}

static float *tg_ensure_opt1(tg_tensor *p) {
    if (!p) {
        return NULL;
    }
    if (p->opt1) {
        return p->opt1;
    }

    size_t n = tg_numel(p);
    p->opt1 = tg_alloc_state_zeroed(n);
    return p->opt1;
}

static float *tg_ensure_opt2(tg_tensor *p) {
    if (!p) {
        return NULL;
    }
    if (p->opt2) {
        return p->opt2;
    }

    size_t n = tg_numel(p);
    p->opt2 = tg_alloc_state_zeroed(n);
    return p->opt2;
}

void tg_param_free(tg_tensor *tensor) {
    if (!tensor) {
        return;
    }

    free(tensor->opt1);
    free(tensor->opt2);
    tensor->opt1 = NULL;
    tensor->opt2 = NULL;

    tg_tensor_destroy(tensor);
}

void tg_params_zero_grad(tg_tensor **params, int n) {
    if (!params || n <= 0) {
        return;
    }

    for (int i = 0; i < n; ++i) {
        tg_tensor *p = params[i];
        if (!p) {
            continue;
        }
        tg_zero_grad(p);
    }
}

void tg_param_list_init(tg_param_list *list) {
    if (!list) {
        return;
    }

    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

void tg_param_list_destroy(tg_param_list *list) {
    if (!list) {
        return;
    }

    free(list->items);
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

tg_status tg_param_list_reserve(tg_param_list *list, int capacity) {
    if (!list || capacity < 0) {
        return TG_ERR_INVALID_ARGUMENT;
    }

    if (capacity <= list->capacity) {
        return TG_OK;
    }

    tg_tensor **new_items = (tg_tensor **)realloc(
        list->items,
        (size_t)capacity * sizeof(tg_tensor *)
    );
    if (!new_items) {
        return TG_ERR_OOM;
    }

    list->items = new_items;
    list->capacity = capacity;
    return TG_OK;
}

tg_status tg_param_list_push(tg_param_list *list, tg_tensor *param) {
    if (!list || !param) {
        return TG_ERR_INVALID_ARGUMENT;
    }

    /*
    Нам важно не допустить попадание временных arena tensors в params.
    */
    if (!tg_param_is_heap_backed(param)) {
        return TG_ERR_INVALID_ARGUMENT;
    }

    if (list->count == list->capacity) {
        int new_capacity = (list->capacity > 0) ? (list->capacity * 2) : 8;
        tg_status st = tg_param_list_reserve(list, new_capacity);
        if (st != TG_OK) {
            return st;
        }
    }

    list->items[list->count++] = param;
    return TG_OK;
}

void tg_param_list_zero_grad(const tg_param_list *list) {
    if (!list || !list->items || list->count <= 0) {
        return;
    }

    tg_params_zero_grad(list->items, list->count);
}

void tg_sgd_step(tg_tensor **params, int n, float lr, float momentum) {
    if (!params || n <= 0) {
        return;
    }
    if (lr == 0.0f) {
        return;
    }

    bool use_momentum = (momentum != 0.0f);

    for (int i = 0; i < n; ++i) {
        tg_tensor *p = params[i];
        if (!tg_param_is_updatable(p)) {
            continue;
        }

        size_t numel = tg_numel(p);
        if (numel == 0) {
            continue;
        }

        if (!use_momentum) {
            for (size_t j = 0; j < numel; ++j) {
                p->data[j] -= lr * p->grad[j];
            }
            continue;
        }

        float *v = tg_ensure_opt1(p); /* velocity */
        if (!v) {
            /*
            Нет status-return, поэтому при OOM просто пропускаем параметр.
            Это безопаснее, чем писать в NULL.
            */
            continue;
        }

        for (size_t j = 0; j < numel; ++j) {
            v[j] = momentum * v[j] - lr * p->grad[j];
            p->data[j] += v[j];
        }
    }
}

void tg_adam_step(
    tg_tensor **params,
    int n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int t
) {
    if (!params || n <= 0) {
        return;
    }
    if (lr == 0.0f) {
        return;
    }
    if (eps <= 0.0f) {
        eps = 1e-8f;
    }
    if (t < 1) {
        t = 1;
    }

    float b1_corr = 1.0f - powf(beta1, (float)t);
    float b2_corr = 1.0f - powf(beta2, (float)t);

    /*
    Если кто-то передал beta1/beta2 ~= 1 и коррекция ушла в 0,
    просто не делаем update во избежание inf.
    */
    if (b1_corr == 0.0f || b2_corr == 0.0f) {
        return;
    }

    for (int i = 0; i < n; ++i) {
        tg_tensor *p = params[i];
        if (!tg_param_is_updatable(p)) {
            continue;
        }

        size_t numel = tg_numel(p);
        if (numel == 0) {
            continue;
        }

        float *m = tg_ensure_opt1(p);
        float *v = tg_ensure_opt2(p);
        if (!m || !v) {
            continue;
        }

        for (size_t j = 0; j < numel; ++j) {
            float g = p->grad[j];

            m[j] = beta1 * m[j] + (1.0f - beta1) * g;
            v[j] = beta2 * v[j] + (1.0f - beta2) * g * g;

            float m_hat = m[j] / b1_corr;
            float v_hat = v[j] / b2_corr;

            p->data[j] -= lr * m_hat / (sqrtf(v_hat) + eps);
        }
    }
}
