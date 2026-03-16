# Python API: tinygradc

## Оглавление

- [Обзор](#обзор)
- [`TensorRef`](#tensorref)
- [`TinyGradC`](#tinygradc)
  - [Runtime](#runtime)
  - [Создание и lifetime tensor](#создание-и-lifetime-tensor)
  - [Чтение и работа с данными](#чтение-и-работа-с-данными)
  - [Операции](#операции)
  - [Функции потерь](#функции-потерь)
  - [Оптимизаторы](#оптимизаторы)
  - [Сохранение и загрузка параметров](#сохранение-и-загрузка-параметров)

## Обзор

Этот файл автоматически генерируется из docstring'ов в `python/tinygradc/core.py`.

Основные классы Python API:

- `TensorRef` — лёгкая ссылка на tensor в C runtime
- `TinyGradC` — основной Python runtime wrapper

Типичный сценарий использования:

```python
from tinygradc import TinyGradC

with TinyGradC() as tg:
    w = tg.param(64, 10, True)
    b = tg.param(1, 10, True)

    x = tg.tensor_from_rows(batch_x, requires_grad=False)
    y = tg.tensor_from_rows(batch_y, requires_grad=False)

    logits = tg.linear(x, w, b)
    loss = tg.softmax_cross_entropy(logits, y)

    tg.backward(loss)
    tg.adam_step([w, b], lr=1e-3, t=1)
    tg.reset()
```

## `TensorRef`

Лёгкая ссылка на tensor, которым управляет tinygradc.

Этот объект хранит:
- handle:
    непрозрачный указатель на C tensor
- backing:
    необязательный Python ctypes-буфер, который нужен, чтобы данные
    оставались живы со стороны Python
- rows, cols:
    необязательная информация о форме tensor на стороне Python

Замечания:
- Временные тензоры обычно создаются в arena-памяти и становятся
  невалидными после TinyGradC.reset().
- Параметры, созданные через TinyGradC.param(), живут на heap
  и существуют до вызова TinyGradC.free_param().

### `TensorRef(handle, backing=None, rows=None, cols=None)`

Создать ссылку на tensor.

Аргументы:
    handle:
        Непрозрачный ctypes-указатель на C tensor.
    backing:
        Необязательный Python-буфер, который должен жить,
        пока tensor используется.
    rows:
        Необязательное число строк.
    cols:
        Необязательное число столбцов.


## `TinyGradC`

Минимальная Python-обёртка над C runtime библиотеки tinygradc.

TinyGradC отвечает за:
- загрузку shared library
- создание и уничтожение arena
- создание trainable parameters
- преобразование Python-списков в временные tensors
- вызов tensor-операций
- вычисление loss-функций
- запуск backward
- шаги оптимизаторов
- сохранение и загрузку параметров

Типичный сценарий использования:
    with TinyGradC() as tg:
        w = tg.param(64, 10, True)
        b = tg.param(1, 10, True)

        x = tg.tensor_from_rows(batch_x, requires_grad=False)
        y = tg.tensor_from_rows(batch_y, requires_grad=False)

        logits = tg.linear(x, w, b)
        loss = tg.softmax_cross_entropy(logits, y)

        tg.backward(loss)
        tg.adam_step([w, b], lr=1e-3, t=1)

        tg.reset()

Важно:
- Временные тензоры создаются в arena и становятся невалидными после reset().
- Heap-backed параметры нужно освобождать явно через free_param().
- close() уничтожает arena, но не освобождает параметры автоматически.

### `TinyGradC(lib_path=None, arena_bytes=8388608)`

Создать runtime TinyGradC.

Аргументы:
    lib_path:
        Необязательный явный путь до shared library tinygradc.
        Если None, библиотека будет найдена автоматически.
    arena_bytes:
        Размер временной arena в байтах.

Исключения:
    FileNotFoundError:
        Если shared library не удалось найти.
    RuntimeError:
        Если не удалось создать arena.

### Runtime

#### `TinyGradC.close(self)`

Уничтожить arena и освободить ресурсы runtime.

Замечания:
- Параметры, созданные через param(), не освобождаются автоматически.
- Heap-backed параметры нужно освобождать через free_param().

#### `TinyGradC.reset(self)`

Сбросить arena-память.

Это делает невалидными все временные тензоры, созданные во время
forward/inference. Обычно reset() вызывается после шага обучения
или после прохода inference, когда временные graph-объекты больше
не нужны.

### Создание и lifetime tensor

#### `TinyGradC.param(self, rows, cols, requires_grad=True)`

Создать долгоживущий параметр модели.

Аргументы:
    rows:
        Число строк.
    cols:
        Число столбцов.
    requires_grad:
        Нужно ли выделить буфер градиента.

Возвращает:
    TensorRef для heap-backed параметра.

Типичные примеры использования:
- веса модели
- bias
- обучаемые параметры

#### `TinyGradC.free_param(self, tensor)`

Освободить параметр, созданный через param().

Аргументы:
    tensor:
        TensorRef, указывающий на heap-backed параметр.

Замечания:
- Безопасно вызывать и для уже освобождённого TensorRef,
  если handle == None.
- После вызова объект TensorRef больше нельзя использовать.

#### `TinyGradC.tensor_from_rows(self, rows_2d, requires_grad=False)`

Обернуть Python 2D-список во временный tensor.

Аргументы:
    rows_2d:
        Вложенный Python-список формы [rows][cols].
    requires_grad:
        Нужно ли выделить буфер градиента.

Возвращает:
    TensorRef для временного tensor.

Замечания:
- Входные данные копируются в ctypes-буфер, которым управляет Python.
- Полученный tensor живёт в arena.
- После reset() такой tensor нужно считать невалидным.

### Чтение и работа с данными

#### `TinyGradC.numel(self, tensor)`

Вернуть число элементов в tensor.

Аргументы:
    tensor:
        Объект TensorRef.

Возвращает:
    Целое число элементов.

#### `TinyGradC.data_1d(self, tensor)`

Прочитать данные tensor как плоский Python-список.

Аргументы:
    tensor:
        Объект TensorRef.

Возвращает:
    Плоский список float-значений.

#### `TinyGradC.data_2d(self, tensor, rows=None, cols=None)`

Прочитать данные tensor как вложенный Python-список.

Аргументы:
    tensor:
        Объект TensorRef.
    rows:
        Необязательное число строк. Если None, используется tensor.rows.
    cols:
        Необязательное число столбцов. Если None, используется tensor.cols.

Возвращает:
    Вложенный список формы [rows][cols].

Замечания:
- Это удобно для отладки, визуализации и demo.
- Для больших tensor'ов такое преобразование может быть медленным.

#### `TinyGradC.scalar(self, tensor)`

Прочитать значение scalar tensor.

Аргументы:
    tensor:
        TensorRef, который должен содержать scalar.

Возвращает:
    Float-значение из tensor[0].

Замечания:
- Обычно используется для loss tensor формы [1 x 1].

#### `TinyGradC.copy_into_param(self, tensor, flat_values)`

Скопировать Python-значения в параметр tensor.

Аргументы:
    tensor:
        TensorRef параметра, в который нужно записать данные.
    flat_values:
        Плоская последовательность float-подобных значений.
        Длина должна совпадать с numel tensor.

Исключения:
    ValueError:
        Если число элементов не совпадает с размером tensor.

#### `TinyGradC.zero_grads(self, params)`

Обнулить градиенты у списка параметров.

Аргументы:
    params:
        Список TensorRef-параметров.

Замечания:
- backward() не обнуляет градиенты автоматически.
- Обычно zero_grads(...) вызывается перед новым шагом обучения.

### Операции

#### `TinyGradC.matmul(self, a, b)`

Матричное умножение.

Формы:
    a: [B x D]
    b: [D x H]

Возвращает:
    TensorRef формы [B x H].

#### `TinyGradC.add(self, a, b)`

Поэлементное сложение.

Поддерживаемые формы:
    - одинаковые формы
    - bias-broadcast для случая [R x C] + [1 x C]

Возвращает:
    TensorRef с результатом.

#### `TinyGradC.relu(self, x)`

Применить ReLU-активацию.

Аргументы:
    x:
        Входной TensorRef.

Возвращает:
    TensorRef той же формы.

#### `TinyGradC.sigmoid(self, x)`

Применить sigmoid-активацию.

Аргументы:
    x:
        Входной TensorRef.

Возвращает:
    TensorRef той же формы.

#### `TinyGradC.tanh(self, x)`

Применить tanh-активацию.

Аргументы:
    x:
        Входной TensorRef.

Возвращает:
    TensorRef той же формы.

#### `TinyGradC.linear(self, x, weight, bias=None)`

Применить линейный слой.

Эквивалентно:
    x @ weight + bias

Формы:
    x:      [B x IN]
    weight: [IN x OUT]
    bias:   None или [1 x OUT]

Возвращает:
    TensorRef формы [B x OUT].

### Функции потерь

#### `TinyGradC.mse(self, pred, target)`

Mean squared error loss.

Формы:
    pred:   [R x C]
    target: [R x C]

Возвращает:
    Scalar TensorRef формы [1 x 1].

#### `TinyGradC.bce_with_logits(self, logits, target)`

Binary cross-entropy loss с logits.

Формы:
    logits: [R x C]
    target: [R x C]

Возвращает:
    Scalar TensorRef формы [1 x 1].

Замечания:
- Используется численно стабильная fused-реализация BCE-with-logits.

#### `TinyGradC.softmax_cross_entropy(self, logits, target)`

Softmax cross-entropy loss.

Формы:
    logits: [B x C]
    target: [B x C], one-hot encoded labels

Возвращает:
    Scalar TensorRef формы [1 x 1].

Замечания:
- Используется стабильная fused-реализация softmax + cross-entropy.

#### `TinyGradC.backward(self, loss)`

Запустить backward, начиная со scalar loss.

Аргументы:
    loss:
        Scalar TensorRef, обычно возвращённый loss-функцией.

Исключения:
    RuntimeError:
        Если backward завершился ошибкой в C runtime.

Замечания:
- Существующие градиенты не обнуляются автоматически.
- Перед новым шагом оптимизации при необходимости вызывай zero_grads(...).

### Оптимизаторы

#### `TinyGradC.sgd_step(self, params, lr, momentum=0.0)`

Выполнить один шаг оптимизатора SGD.

Аргументы:
    params:
        Список TensorRef-параметров.
    lr:
        Learning rate.
    momentum:
        Коэффициент momentum. Если 0.0, используется обычный SGD.

#### `TinyGradC.adam_step(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-08, t=1)`

Выполнить один шаг оптимизатора Adam.

Аргументы:
    params:
        Список TensorRef-параметров.
    lr:
        Learning rate.
    beta1:
        Коэффициент затухания для первой оценки момента.
    beta2:
        Коэффициент затухания для второй оценки момента.
    eps:
        Маленькая константа для численной стабильности.
    t:
        Номер шага, начиная с 1.

### Сохранение и загрузка параметров

#### `TinyGradC.params_save(self, path, params)`

Сохранить параметры модели на диск.

Аргументы:
    path:
        Путь к выходному файлу.
    params:
        Список TensorRef-параметров.

Исключения:
    RuntimeError:
        Если сохранение завершилось ошибкой в C runtime.

Замечания:
- Используется бинарный формат параметров tinygradc.
- Предназначено для сохранения весов модели.

#### `TinyGradC.params_load(self, path, params)`

Загрузить параметры модели с диска.

Аргументы:
    path:
        Путь к входному файлу.
    params:
        Список уже созданных TensorRef-параметров.

Исключения:
    RuntimeError:
        Если загрузка завершилась ошибкой в C runtime.

Замечания:
- Формы параметров должны совпадать с тем, что записано в файле.
- Градиенты и состояние оптимизатора при загрузке сбрасываются
  на стороне C runtime.
