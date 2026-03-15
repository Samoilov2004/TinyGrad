#include "tg.h"

#include <stdlib.h>

struct tg_tensor {
    size_t size;
    float *data;
};

const char *tg_version_string(void) {
    return "tinygradc 0.1.0";
}

tg_tensor *tg_tensor_create(size_t size) {
    tg_tensor *tensor = (tg_tensor *)calloc(1, sizeof(*tensor));
    if (tensor == NULL) {
        return NULL;
    }

    tensor->size = size;

    if (size > 0U) {
        tensor->data = (float *)calloc(size, sizeof(*tensor->data));
        if (tensor->data == NULL) {
            free(tensor);
            return NULL;
        }
    }

    return tensor;
}

void tg_tensor_destroy(tg_tensor *tensor) {
    if (tensor == NULL) {
        return;
    }

    free(tensor->data);
    free(tensor);
}

size_t tg_tensor_size(const tg_tensor *tensor) {
    return tensor != NULL ? tensor->size : 0U;
}

float *tg_tensor_data(tg_tensor *tensor) {
    return tensor != NULL ? tensor->data : NULL;
}

const float *tg_tensor_data_const(const tg_tensor *tensor) {
    return tensor != NULL ? tensor->data : NULL;
}

tg_status tg_backward(tg_tensor *loss) {
    (void)loss;
    return TG_ERR_UNIMPLEMENTED;
}
