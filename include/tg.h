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

#define TG_OP_MAX_INPUTS 3

typedef enum tg_status {
    TG_OK = 0,
    TG_ERR_INVALID_ARGUMENT = 1,
    TG_ERR_OOM = 2,
    TG_ERR_UNIMPLEMENTED = 3
} tg_status;

typedef struct tg_arena tg_arena;
typedef struct tg_tensor tg_tensor;
typedef struct tg_op tg_op;

/*
Простой список параметров модели.
Хранит только массив указателей на tg_tensor, сами tensor не освобождает.
*/
typedef struct tg_param_list {
    tg_tensor **items;
    int count;
    int capacity;
} tg_param_list;

/*
Backward callback contract
--------------------------
- out              : выход операции; out->grad уже содержит dL/d(out)
- op->inputs[i]    : входы операции
- callback должен накапливать вклад во входные градиенты через +=
*/
typedef void (*tg_backward_fn)(tg_op *op, tg_tensor *out);

/*
Контекст операции динамического графа.
Обычно создаётся в arena и привязывается к результирующему tensor через out->op.
*/
struct tg_op {
    tg_tensor *inputs[TG_OP_MAX_INPUTS];
    int num_inputs;
    tg_backward_fn backward;

    /* Небольшой generic cache под параметры операции. */
    int aux_i32[4];
    float aux_f32[4];
    void *aux_ptr;
};

/*
Базовый 2D tensor float32.

Публичные поля:
- data            : указатель на данные размера rows * cols
- grad            : буфер градиента того же размера или NULL
- rows, cols      : форма 2D
- requires_grad   : нужно ли хранить/накапливать градиент
- op              : op-node, породивший tensor; NULL для leaf tensor

Optimizer state:
- opt1            : опциональный state-buffer оптимизатора (heap), напр. velocity или m
- opt2            : опциональный state-buffer оптимизатора (heap), напр. v

Служебные поля владения:
- owns_data       : tg_tensor_destroy() освобождает data, если true
- owns_grad       : tg_tensor_destroy() освобождает grad, если true
- owns_self       : tg_tensor_destroy() освобождает сам объект tensor, если true
*/
struct tg_tensor {
    float *data;
    float *grad;
    int rows;
    int cols;
    bool requires_grad;

    tg_op *op;

    float *opt1;
    float *opt2;

    bool owns_data;
    bool owns_grad;
    bool owns_self;
};

const char *tg_version_string(void);

/*
Новый tensor API
----------------

1) Долгоживущий параметр на heap.
   - tg_tensor              : библиотека / tg_tensor_destroy() или tg_param_free()
   - data                   : библиотека / tg_tensor_destroy() или tg_param_free()
   - grad (если есть)       : библиотека / tg_tensor_destroy() или tg_param_free()
   - opt1/opt2              : библиотека / tg_param_free()
*/
tg_tensor *tg_param_create(int rows, int cols, bool requires_grad);

/*
2) Временный tensor внутри arena.
   - tg_tensor              : arena
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

   - tg_tensor              : arena
   - data                   : внешний код, библиотека не владеет
   - grad (опционально)     : arena

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
Низкоуровневый API для op-node
------------------------------
Обычно используется реализацией операций.
*/
tg_op *tg_op_create(tg_arena *arena, int num_inputs, tg_backward_fn backward);
tg_status tg_op_set_input(tg_op *op, int index, tg_tensor *input);

/*
Базовые операции
----------------
Все операции:
- создают output tensor в arena
- возвращают NULL при некорректных аргументах или OOM
- создают op-node только если хотя бы один вход requires_grad == true

Поддержка форм:
- tg_add / tg_sub:
    * одинаковые формы [R x C] и [R x C]
    * частичный bias-broadcast: b может быть [1 x C], тогда out = [R x C]
- tg_mul:
    * только одинаковые формы
- tg_matmul:
    * a: [B x D], b: [D x H] -> out: [B x H]
- tg_sum / tg_mean:
    * x: [R x C] -> out: [1 x 1]
*/
tg_tensor *tg_add(tg_arena *arena, tg_tensor *a, tg_tensor *b);
tg_tensor *tg_sub(tg_arena *arena, tg_tensor *a, tg_tensor *b);
tg_tensor *tg_mul(tg_arena *arena, tg_tensor *a, tg_tensor *b);
tg_tensor *tg_matmul(tg_arena *arena, tg_tensor *a, tg_tensor *b);
tg_tensor *tg_sum(tg_arena *arena, tg_tensor *x);
tg_tensor *tg_mean(tg_arena *arena, tg_tensor *x);

/*
Нелинейности
------------
Все операции:
- x: [R x C] -> out: [R x C]
- backward использует:
    * tg_relu     : output tensor (mask через out->data > 0)
    * tg_sigmoid  : output tensor, s = out->data
    * tg_tanh     : output tensor, t = out->data
*/
tg_tensor *tg_relu(tg_arena *arena, tg_tensor *x);
tg_tensor *tg_sigmoid(tg_arena *arena, tg_tensor *x);
tg_tensor *tg_tanh(tg_arena *arena, tg_tensor *x);

/*
Loss функции
------------
Target tensors трактуются как константы:
градиенты накапливаются только в pred/logits, но не в target.

- tg_mse:
    * pred, target: одинаковая форма [R x C]
    * returns: scalar [1 x 1]

- tg_bce_with_logits:
    * logits, target: одинаковая форма [R x C]
    * returns: scalar [1 x 1]

- tg_softmax_cross_entropy:
    * logits: [B x C]
    * target_onehot: [B x C]
    * returns: scalar [1 x 1]
*/
tg_tensor *tg_mse(tg_arena *arena, tg_tensor *pred, tg_tensor *target);
tg_tensor *tg_bce_with_logits(tg_arena *arena, tg_tensor *logits, tg_tensor *target);
tg_tensor *tg_softmax_cross_entropy(
    tg_arena *arena,
    tg_tensor *logits,
    tg_tensor *target_onehot
);

/*
Параметры и оптимизаторы
------------------------
Важно:
- в список параметров должны попадать только heap-параметры, а не временные arena tensors
- tg_param_list_push() это проверяет
- tg_param_free() освобождает optimizer state + сам tensor

Optimizer state convention:
- SGD + momentum:
    opt1 = velocity
- Adam:
    opt1 = m
    opt2 = v
*/
void tg_param_free(tg_tensor *tensor);

void tg_params_zero_grad(tg_tensor **params, int n);

void tg_param_list_init(tg_param_list *list);
void tg_param_list_destroy(tg_param_list *list);
tg_status tg_param_list_reserve(tg_param_list *list, int capacity);
tg_status tg_param_list_push(tg_param_list *list, tg_tensor *param);
void tg_param_list_zero_grad(const tg_param_list *list);

void tg_sgd_step(tg_tensor **params, int n, float lr, float momentum);
void tg_adam_step(
    tg_tensor **params,
    int n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int t
);

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

Примечание:
- для trainable heap parameters лучше использовать tg_param_free(),
  так как он также чистит optimizer state.
*/
void tg_tensor_destroy(tg_tensor *tensor);

/* Совместимость со старым API: эквивалент tg_numel(). */
size_t tg_tensor_size(const tg_tensor *tensor);

float *tg_tensor_data(tg_tensor *tensor);
const float *tg_tensor_data_const(const tg_tensor *tensor);

/*
Запуск обратного прохода.

Текущее MVP-предположение:
- loss должен быть scalar tensor (numel == 1)
- existing grad buffers не обнуляются автоматически
- seed: loss->grad[0] = 1.0f
*/
tg_status tg_backward(tg_tensor *loss);

/*
Arena allocator.
align должен быть степенью двойки; align == 0 означает выравнивание по умолчанию.
*/
tg_arena *tg_arena_create(size_t initial_bytes);
void tg_arena_destroy(tg_arena *arena);
void tg_arena_reset(tg_arena *arena);
void *tg_arena_alloc(tg_arena *arena, size_t size, size_t align);

#ifdef __cplusplus
}
#endif

#endif /* TG_H */
