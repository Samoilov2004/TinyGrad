#ifndef TG_H
#define TG_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

#define TG_VERSION_MAJOR 0
#define TG_VERSION_MINOR 1
#define TG_VERSION_PATCH 0

typedef enum tg_status {
    TG_OK = 0,
    TG_ERR_INVALID_ARGUMENT = 1,
    TG_ERR_OOM = 2,
    TG_ERR_UNIMPLEMENTED = 3
} tg_status;

typedef struct tg_tensor tg_tensor;

const char *tg_version_string(void);

tg_tensor *tg_tensor_create(size_t size);
void tg_tensor_destroy(tg_tensor *tensor);

size_t tg_tensor_size(const tg_tensor *tensor);
float *tg_tensor_data(tg_tensor *tensor);
const float *tg_tensor_data_const(const tg_tensor *tensor);

tg_status tg_backward(tg_tensor *loss);

#ifdef __cplusplus
}
#endif

#endif /* TG_H */
