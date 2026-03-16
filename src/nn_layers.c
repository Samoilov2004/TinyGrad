#include "tg.h"

tg_tensor *tg_linear(
    tg_arena *arena,
    tg_tensor *x,
    tg_tensor *weight,
    tg_tensor *bias
) {
    tg_tensor *out;

    if (!arena || !x || !weight) {
        return NULL;
    }

    out = tg_matmul(arena, x, weight);
    if (!out) {
        return NULL;
    }

    if (bias) {
        out = tg_add(arena, out, bias);
        if (!out) {
            return NULL;
        }
    }

    return out;
}
