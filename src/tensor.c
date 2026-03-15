#include "tg.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static bool tg_shape_numel(int rows, int cols, size_t *out_numel) {
    size_t r;
    size_t c;

    if (out_numel == NULL) {
        return false;
    }

    if (rows < 0 || cols < 0) {
        return false;
    }

    r = (size_t)rows;
    c = (size_t)cols;

    if (c != 0 && r > (SIZE_MAX / c)) {
        return false;
    }

    *out_numel = r * c;
    return true;
}

static bool tg_numel_bytes(size_t numel, size_t *out_bytes) {
    if (out_bytes == NULL) {
        return false;
    }

    if (numel > (SIZE_MAX / sizeof(float))) {
        return false;
    }

    *out_bytes = numel * sizeof(float);
    return true;
}

static tg_tensor *tg_tensor_init(
    tg_tensor *tensor,
    float *data,
    float *grad,
    int rows,
    int cols,
    bool requires_grad,
    bool owns_data,
    bool owns_grad,
    bool owns_self
) {
    if (tensor == NULL) {
        return NULL;
    }

    tensor->data = data;
    tensor->grad = grad;
    tensor->rows = rows;
    tensor->cols = cols;
    tensor->requires_grad = requires_grad;

    /*
    Leaf tensor по умолчанию не имеет op.
    Если tensor является результатом операции, out->op потом заполнит код op.
    */
    tensor->op = NULL;

    /*
    Optimizer state живёт только у heap-параметров по необходимости,
    но безопаснее всегда явно инициализировать в NULL.
    */
    tensor->opt1 = NULL;
    tensor->opt2 = NULL;

    tensor->owns_data = owns_data;
    tensor->owns_grad = owns_grad;
    tensor->owns_self = owns_self;

    return tensor;
}

static float *tg_heap_alloc_zeroed_f32(size_t numel) {
    size_t bytes;

    if (numel == 0) {
        return NULL;
    }

    if (!tg_numel_bytes(numel, &bytes)) {
        return NULL;
    }

    (void)bytes; /* calloc уже принимает count/size, bytes только для проверки overflow */
    return (float *)calloc(numel, sizeof(float));
}

static float *tg_arena_alloc_zeroed_f32(tg_arena *arena, size_t numel) {
    float *ptr;
    size_t bytes;

    if (arena == NULL) {
        return NULL;
    }

    if (numel == 0) {
        return NULL;
    }

    if (!tg_numel_bytes(numel, &bytes)) {
        return NULL;
    }

    ptr = (float *)tg_arena_alloc(arena, bytes, _Alignof(float));
    if (ptr == NULL) {
        return NULL;
    }

    memset(ptr, 0, bytes);
    return ptr;
}

tg_tensor *tg_param_create(int rows, int cols, bool requires_grad) {
    tg_tensor *tensor;
    float *data = NULL;
    float *grad = NULL;
    size_t numel = 0;

    if (!tg_shape_numel(rows, cols, &numel)) {
        return NULL;
    }

    tensor = (tg_tensor *)calloc(1, sizeof(*tensor));
    if (tensor == NULL) {
        return NULL;
    }

    data = tg_heap_alloc_zeroed_f32(numel);
    if (numel > 0 && data == NULL) {
        free(tensor);
        return NULL;
    }

    if (requires_grad) {
        grad = tg_heap_alloc_zeroed_f32(numel);
        if (numel > 0 && grad == NULL) {
            free(data);
            free(tensor);
            return NULL;
        }
    }

    return tg_tensor_init(
        tensor,
        data,
        grad,
        rows,
        cols,
        requires_grad,
        true,
        true,
        true
    );
}

tg_tensor *tg_tensor_tmp(tg_arena *arena, int rows, int cols, bool requires_grad) {
    tg_tensor *tensor;
    float *data = NULL;
    float *grad = NULL;
    size_t numel = 0;

    if (arena == NULL) {
        return NULL;
    }

    if (!tg_shape_numel(rows, cols, &numel)) {
        return NULL;
    }

    data = tg_arena_alloc_zeroed_f32(arena, numel);
    if (numel > 0 && data == NULL) {
        return NULL;
    }

    if (requires_grad) {
        grad = tg_arena_alloc_zeroed_f32(arena, numel);
        if (numel > 0 && grad == NULL) {
            return NULL;
        }
    }

    tensor = (tg_tensor *)tg_arena_alloc(arena, sizeof(*tensor), _Alignof(tg_tensor));
    if (tensor == NULL) {
        return NULL;
    }

    memset(tensor, 0, sizeof(*tensor));

    return tg_tensor_init(
        tensor,
        data,
        grad,
        rows,
        cols,
        requires_grad,
        false,
        false,
        false
    );
}

tg_tensor *tg_tensor_from_buffer(
    tg_arena *arena,
    float *ptr,
    int rows,
    int cols,
    bool requires_grad
) {
    tg_tensor *tensor;
    float *grad = NULL;
    size_t numel = 0;

    if (arena == NULL) {
        return NULL;
    }

    if (!tg_shape_numel(rows, cols, &numel)) {
        return NULL;
    }

    if (numel > 0 && ptr == NULL) {
        return NULL;
    }

    if (requires_grad) {
        grad = tg_arena_alloc_zeroed_f32(arena, numel);
        if (numel > 0 && grad == NULL) {
            return NULL;
        }
    }

    tensor = (tg_tensor *)tg_arena_alloc(arena, sizeof(*tensor), _Alignof(tg_tensor));
    if (tensor == NULL) {
        return NULL;
    }

    memset(tensor, 0, sizeof(*tensor));

    return tg_tensor_init(
        tensor,
        ptr,
        grad,
        rows,
        cols,
        requires_grad,
        false,
        false,
        false
    );
}

void tg_zero_grad(tg_tensor *tensor) {
    size_t numel;
    size_t bytes;

    if (tensor == NULL || tensor->grad == NULL) {
        return;
    }

    numel = tg_numel(tensor);
    if (numel == 0) {
        return;
    }

    if (!tg_numel_bytes(numel, &bytes)) {
        return;
    }

    memset(tensor->grad, 0, bytes);
}

size_t tg_numel(const tg_tensor *tensor) {
    size_t numel = 0;

    if (tensor == NULL) {
        return 0;
    }

    if (!tg_shape_numel(tensor->rows, tensor->cols, &numel)) {
        return 0;
    }

    return numel;
}

tg_tensor *tg_tensor_create(size_t size) {
    if (size > (size_t)INT_MAX) {
        return NULL;
    }

    return tg_param_create(1, (int)size, false);
}

void tg_tensor_destroy(tg_tensor *tensor) {
    if (tensor == NULL) {
        return;
    }

    /*
    Optimizer state всегда heap-allocated для долгоживущих параметров.
    Для arena tensors тут обычно NULL; free(NULL) безопасен.
    */
    free(tensor->opt1);
    free(tensor->opt2);
    tensor->opt1 = NULL;
    tensor->opt2 = NULL;

    if (tensor->owns_grad) {
        free(tensor->grad);
        tensor->grad = NULL;
    }

    if (tensor->owns_data) {
        free(tensor->data);
        tensor->data = NULL;
    }

    if (tensor->owns_self) {
        free(tensor);
    }
}

size_t tg_tensor_size(const tg_tensor *tensor) {
    return tg_numel(tensor);
}

float *tg_tensor_data(tg_tensor *tensor) {
    if (tensor == NULL) {
        return NULL;
    }

    return tensor->data;
}

const float *tg_tensor_data_const(const tg_tensor *tensor) {
    if (tensor == NULL) {
        return NULL;
    }

    return tensor->data;
}
