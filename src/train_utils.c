#include "tg.h"

#include <math.h>
#include <stddef.h>

static int tg_param_has_grad(const tg_tensor *p) {
    return p && p->data && p->grad && p->requires_grad;
}

void tg_add_l2_grad(tg_tensor **params, int n, float weight_decay) {
    int i;

    if (!params || n <= 0 || weight_decay == 0.0f) {
        return;
    }

    for (i = 0; i < n; ++i) {
        tg_tensor *p = params[i];
        size_t j;
        size_t numel;

        if (!tg_param_has_grad(p)) {
            continue;
        }

        numel = tg_numel(p);
        for (j = 0; j < numel; ++j) {
            p->grad[j] += weight_decay * p->data[j];
        }
    }
}

float tg_grad_global_norm(tg_tensor **params, int n) {
    int i;
    double acc = 0.0;

    if (!params || n <= 0) {
        return 0.0f;
    }

    for (i = 0; i < n; ++i) {
        tg_tensor *p = params[i];
        size_t j;
        size_t numel;

        if (!tg_param_has_grad(p)) {
            continue;
        }

        numel = tg_numel(p);
        for (j = 0; j < numel; ++j) {
            double g = (double)p->grad[j];
            acc += g * g;
        }
    }

    return (float)sqrt(acc);
}

void tg_clip_grad_norm(tg_tensor **params, int n, float max_norm) {
    int i;
    float norm;
    float scale;

    if (!params || n <= 0 || max_norm <= 0.0f) {
        return;
    }

    norm = tg_grad_global_norm(params, n);
    if (norm <= max_norm || norm <= 1e-12f) {
        return;
    }

    scale = max_norm / (norm + 1e-12f);

    for (i = 0; i < n; ++i) {
        tg_tensor *p = params[i];
        size_t j;
        size_t numel;

        if (!tg_param_has_grad(p)) {
            continue;
        }

        numel = tg_numel(p);
        for (j = 0; j < numel; ++j) {
            p->grad[j] *= scale;
        }
    }
}

void tg_clip_grad_value(tg_tensor **params, int n, float clip_value) {
    int i;
    float lo;
    float hi;

    if (!params || n <= 0 || clip_value <= 0.0f) {
        return;
    }

    lo = -clip_value;
    hi = clip_value;

    for (i = 0; i < n; ++i) {
        tg_tensor *p = params[i];
        size_t j;
        size_t numel;

        if (!tg_param_has_grad(p)) {
            continue;
        }

        numel = tg_numel(p);
        for (j = 0; j < numel; ++j) {
            float g = p->grad[j];
            if (g < lo) {
                g = lo;
            } else if (g > hi) {
                g = hi;
            }
            p->grad[j] = g;
        }
    }
}
