#ifndef TG_H
#define TG_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TG_VERSION_MAJOR 0
#define TG_VERSION_MINOR 1
#define TG_VERSION_PATCH 0

typedef enum tg_status {
    TG_OK = 0,
    TG_ERR_INVALID_ARGUMENT = 1,
    TG_ERR_OOM = 2,
    TG_ERR_UNIMPLEMENTED = 3
} tg_status;

typedef struct tg_arena tg_arena;

/*
Базовый 2D tensor float32.

Публичные поля:
- data            : указатель на данные размера rows * cols
- grad            : буфер градиента того же размера или NULL
- rows, cols      : форма 2D
- requires_grad   : нужно ли хранить/накапливать градиент

Служебные поля владения:
- owns_data       : tg_tensor_destroy() освобождает data, если true
- owns_grad       : tg_tensor_destroy() освобождает grad, если true
- owns_self       : tg_tensor_destroy() освобождает сам объект tensor, если true

Важно:
пользователь может читать основные поля, но ownership-флаги лучше считать
внутренними и не менять вручную.
*/
typedef struct tg_tensor {
    float *data;
    float *grad;
    int rows;
    int cols;
    bool requires_grad;

    bool owns_data;
    bool owns_grad;
    bool owns_self;
} tg_tensor;

const char *tg_version_string(void);

/*
Новый tensor API
----------------

1) Долгоживущий параметр на heap.
   - tg_tensor          : библиотека / tg_tensor_destroy()
   - data                   : библиотека / tg_tensor_destroy()
   - grad (если есть)       : библиотека / tg_tensor_destroy()
*/
tg_tensor *tg_param_create(int rows, int cols, bool requires_grad);

/*
2) Временный tensor внутри arena.
   - tg_tensor          : arena
   - data                   : arena
   - grad (если есть)       : arena

   После tg_arena_reset() или tg_arena_destroy() tensor, data и grad
   становятся недействительными.
   tg_tensor_destroy() для такого tensor ничего не освобождает поэлементно.
*/
tg_tensor *tg_tensor_tmp(tg_arena *arena, int rows, int cols, bool requires_grad);

/*
3) Tensor-view поверх внешнего буфера, без владения data.
   Данные НЕ копируются.

   - tg_tensor          : arena
   - data                   : внешний код, библиотека не владеет
   - grad (опционально)       : arena

   После tg_arena_reset() или tg_arena_destroy() сам tensor и grad
   становятся недействительными.
   Буфер data должен жить дольше, чем используется этот view.
*/
tg_tensor *tg_tensor_from_buffer(
    tg_arena *arena,
    float *ptr,
    int rows,
    int cols,
    bool requires_grad
);

/* Обнулить буфер градиента, если он существует. */
void tg_zero_grad(tg_tensor *tensor);

/* Количество элементов: rows * cols. Для NULL/некорректной формы возвращает 0. */
size_t tg_numel(const tg_tensor *tensor);

/*
Совместимость со старым API
---------------------------
Создаёт heap-tensor формы 1 x size без grad.
*/
tg_tensor *tg_tensor_create(size_t size);

/*
Освобождает только те части tensor, которыми он владеет.
Для arena-backed tensor это фактически no-op.
Вроде как безопасно для NULL.
*/
void tg_tensor_destroy(tg_tensor *tensor);

/* Совместимость со старым API: эквивалент tg_numel(). */
size_t tg_tensor_size(const tg_tensor *tensor);

float *tg_tensor_data(tg_tensor *tensor);
const float *tg_tensor_data_const(const tg_tensor *tensor);

tg_status tg_backward(tg_tensor *loss);

/*
Выделитель областей для временных графических объектов.
выравнивание должно быть в степени двойки; align == 0 означает выравнивание по умолчанию
*/
tg_arena *tg_arena_create(size_t initial_bytes);
void tg_arena_destroy(tg_arena *arena);
void tg_arena_reset(tg_arena *arena);
void *tg_arena_alloc(tg_arena *arena, size_t size, size_t align);

#ifdef __cplusplus
}
#endif

#endif /* TG_H */
