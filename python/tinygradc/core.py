"""
Python-обёртка для tinygradc.

Этот модуль предоставляет минимальный высокоуровневый интерфейс к C runtime
библиотеки tinygradc через ctypes.

Основная точка входа:
    TinyGradC

Основные сущности:
- TensorRef:
    лёгкий Python-объект, который хранит ссылку на C tensor
- TinyGradC:
    объект runtime, который управляет arena-памятью и предоставляет доступ
    к операциям над тензорами, loss-функциям, backward, оптимизаторам
    и сохранению / загрузке параметров

Типичный сценарий использования:
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

Важные замечания по lifetime:
- Временные тензоры, созданные во время forward, живут в arena.
- После reset() временные тензоры считаются невалидными.
- Параметры, созданные через param(), живут долго и должны явно освобождаться
  через free_param().
"""

import ctypes
import os
from pathlib import Path


class TensorRef:
    """
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
    """

    def __init__(self, handle, backing=None, rows=None, cols=None):
        """
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
        """
        self.handle = handle
        self.backing = backing
        self.rows = rows
        self.cols = cols


def _find_library():
    """
    Найти shared library tinygradc.

    Порядок поиска:
    1. Переменная окружения TINYGRADC_LIB
    2. Shared library внутри установленного Python-пакета
    3. Shared library в локальной папке build/

    Возвращает:
        Абсолютный путь до shared library.

    Исключения:
        FileNotFoundError:
            Если shared library не удалось найти.
    """
    env = os.environ.get("TINYGRADC_LIB")
    if env and Path(env).exists():
        return str(Path(env).resolve())

    here = Path(__file__).resolve().parent
    candidates = [
        here / "libtinygradc.so",
        here / "libtinygradc.dylib",
        here / "tinygradc.dll",
        here.parent.parent / "build" / "libtinygradc.so",
        here.parent.parent / "build" / "libtinygradc.dylib",
        here.parent.parent / "build" / "tinygradc.dll",
    ]

    for p in candidates:
        if p.exists():
            return str(p.resolve())

    raise FileNotFoundError(
        "tinygradc shared library not found. "
        "Set TINYGRADC_LIB or build the project first."
    )


class TinyGradC:
    """
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
    """

    def __init__(self, lib_path=None, arena_bytes=8 * 1024 * 1024):
        """
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
        """
        if lib_path is None:
            lib_path = _find_library()

        self.lib = ctypes.CDLL(lib_path)

        c_void_p = ctypes.c_void_p
        c_int = ctypes.c_int
        c_float = ctypes.c_float
        c_bool = ctypes.c_bool
        c_size_t = ctypes.c_size_t
        float_p = ctypes.POINTER(c_float)

        self.c_void_p = c_void_p
        self.c_int = c_int
        self.c_float = c_float
        self.c_bool = c_bool
        self.c_size_t = c_size_t
        self.float_p = float_p

        # arena
        self.lib.tg_arena_create.argtypes = [c_size_t]
        self.lib.tg_arena_create.restype = c_void_p

        self.lib.tg_arena_destroy.argtypes = [c_void_p]
        self.lib.tg_arena_destroy.restype = None

        self.lib.tg_arena_reset.argtypes = [c_void_p]
        self.lib.tg_arena_reset.restype = None

        # tensor / params
        self.lib.tg_param_create.argtypes = [c_int, c_int, c_bool]
        self.lib.tg_param_create.restype = c_void_p

        self.lib.tg_param_free.argtypes = [c_void_p]
        self.lib.tg_param_free.restype = None

        self.lib.tg_tensor_from_buffer.argtypes = [c_void_p, float_p, c_int, c_int, c_bool]
        self.lib.tg_tensor_from_buffer.restype = c_void_p

        self.lib.tg_tensor_data.argtypes = [c_void_p]
        self.lib.tg_tensor_data.restype = float_p

        self.lib.tg_numel.argtypes = [c_void_p]
        self.lib.tg_numel.restype = c_size_t

        self.lib.tg_zero_grad.argtypes = [c_void_p]
        self.lib.tg_zero_grad.restype = None

        # ops
        self.lib.tg_matmul.argtypes = [c_void_p, c_void_p, c_void_p]
        self.lib.tg_matmul.restype = c_void_p

        self.lib.tg_add.argtypes = [c_void_p, c_void_p, c_void_p]
        self.lib.tg_add.restype = c_void_p

        self.lib.tg_relu.argtypes = [c_void_p, c_void_p]
        self.lib.tg_relu.restype = c_void_p

        self.lib.tg_sigmoid.argtypes = [c_void_p, c_void_p]
        self.lib.tg_sigmoid.restype = c_void_p

        self.lib.tg_tanh.argtypes = [c_void_p, c_void_p]
        self.lib.tg_tanh.restype = c_void_p

        self.lib.tg_linear.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p]
        self.lib.tg_linear.restype = c_void_p

        # losses
        self.lib.tg_mse.argtypes = [c_void_p, c_void_p, c_void_p]
        self.lib.tg_mse.restype = c_void_p

        self.lib.tg_bce_with_logits.argtypes = [c_void_p, c_void_p, c_void_p]
        self.lib.tg_bce_with_logits.restype = c_void_p

        self.lib.tg_softmax_cross_entropy.argtypes = [c_void_p, c_void_p, c_void_p]
        self.lib.tg_softmax_cross_entropy.restype = c_void_p

        # autograd
        self.lib.tg_backward.argtypes = [c_void_p]
        self.lib.tg_backward.restype = c_int

        # optim
        self.lib.tg_sgd_step.argtypes = [
            ctypes.POINTER(c_void_p), c_int, c_float, c_float
        ]
        self.lib.tg_sgd_step.restype = None

        self.lib.tg_adam_step.argtypes = [
            ctypes.POINTER(c_void_p), c_int, c_float, c_float, c_float, c_float, c_int
        ]
        self.lib.tg_adam_step.restype = None

        # io
        self.lib.tg_params_save.argtypes = [ctypes.c_char_p, ctypes.POINTER(c_void_p), c_int]
        self.lib.tg_params_save.restype = c_int

        self.lib.tg_params_load.argtypes = [ctypes.c_char_p, ctypes.POINTER(c_void_p), c_int]
        self.lib.tg_params_load.restype = c_int

        self.arena = self.lib.tg_arena_create(arena_bytes)
        if not self.arena:
            raise RuntimeError("tg_arena_create failed")

    def __enter__(self):
        """
        Войти в context manager.

        Возвращает:
            Текущий объект TinyGradC.
        """
        return self

    def __exit__(self, exc_type, exc, tb):
        """
        Выйти из context manager и освободить arena-ресурсы.
        """
        self.close()

    def close(self):
        """
        Уничтожить arena и освободить ресурсы runtime.

        Замечания:
        - Параметры, созданные через param(), не освобождаются автоматически.
        - Heap-backed параметры нужно освобождать через free_param().
        """
        if self.arena:
            self.lib.tg_arena_destroy(self.arena)
            self.arena = None

    def reset(self):
        """
        Сбросить arena-память.

        Это делает невалидными все временные тензоры, созданные во время
        forward/inference. Обычно reset() вызывается после шага обучения
        или после прохода inference, когда временные graph-объекты больше
        не нужны.
        """
        self.lib.tg_arena_reset(self.arena)

    def param(self, rows, cols, requires_grad=True):
        """
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
        """
        h = self.lib.tg_param_create(rows, cols, requires_grad)
        if not h:
            raise RuntimeError("tg_param_create failed")
        return TensorRef(h, None, rows, cols)

    def free_param(self, tensor):
        """
        Освободить параметр, созданный через param().

        Аргументы:
            tensor:
                TensorRef, указывающий на heap-backed параметр.

        Замечания:
        - Безопасно вызывать и для уже освобождённого TensorRef,
          если handle == None.
        - После вызова объект TensorRef больше нельзя использовать.
        """
        if tensor and tensor.handle:
            self.lib.tg_param_free(tensor.handle)
            tensor.handle = None
            tensor.backing = None

    def tensor_from_rows(self, rows_2d, requires_grad=False):
        """
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
        """
        rows = len(rows_2d)
        if rows == 0:
            raise ValueError("empty rows")
        cols = len(rows_2d[0])

        flat = []
        for row in rows_2d:
            if len(row) != cols:
                raise ValueError("ragged rows")
            flat.extend(float(x) for x in row)

        buf = (self.c_float * (rows * cols))(*flat)
        h = self.lib.tg_tensor_from_buffer(self.arena, buf, rows, cols, requires_grad)
        if not h:
            raise RuntimeError("tg_tensor_from_buffer failed")

        return TensorRef(h, buf, rows, cols)

    def numel(self, tensor):
        """
        Вернуть число элементов в tensor.

        Аргументы:
            tensor:
                Объект TensorRef.

        Возвращает:
            Целое число элементов.
        """
        return int(self.lib.tg_numel(tensor.handle))

    def data_1d(self, tensor):
        """
        Прочитать данные tensor как плоский Python-список.

        Аргументы:
            tensor:
                Объект TensorRef.

        Возвращает:
            Плоский список float-значений.
        """
        n = self.numel(tensor)
        ptr = self.lib.tg_tensor_data(tensor.handle)
        return [float(ptr[i]) for i in range(n)]

    def data_2d(self, tensor, rows=None, cols=None):
        """
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
        """
        if rows is None:
            rows = tensor.rows
        if cols is None:
            cols = tensor.cols
        ptr = self.lib.tg_tensor_data(tensor.handle)

        out = []
        k = 0
        for _ in range(rows):
            row = []
            for _ in range(cols):
                row.append(float(ptr[k]))
                k += 1
            out.append(row)
        return out

    def scalar(self, tensor):
        """
        Прочитать значение scalar tensor.

        Аргументы:
            tensor:
                TensorRef, который должен содержать scalar.

        Возвращает:
            Float-значение из tensor[0].

        Замечания:
        - Обычно используется для loss tensor формы [1 x 1].
        """
        ptr = self.lib.tg_tensor_data(tensor.handle)
        return float(ptr[0])

    def copy_into_param(self, tensor, flat_values):
        """
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
        """
        n = self.numel(tensor)
        if len(flat_values) != n:
            raise ValueError("size mismatch")
        ptr = self.lib.tg_tensor_data(tensor.handle)
        for i, v in enumerate(flat_values):
            ptr[i] = float(v)

    def zero_grads(self, params):
        """
        Обнулить градиенты у списка параметров.

        Аргументы:
            params:
                Список TensorRef-параметров.

        Замечания:
        - backward() не обнуляет градиенты автоматически.
        - Обычно zero_grads(...) вызывается перед новым шагом обучения.
        """
        for p in params:
            self.lib.tg_zero_grad(p.handle)

    def _arr_params(self, params):
        """
        Преобразовать Python-список TensorRef в ctypes-массив.

        Аргументы:
            params:
                Список TensorRef-параметров.

        Возвращает:
            ctypes-массив сырых tensor handle.

        Замечания:
        - Это внутренний helper, который используется в оптимизаторах
          и функциях сохранения / загрузки.
        """
        return (self.c_void_p * len(params))(*[p.handle for p in params])

    def sgd_step(self, params, lr, momentum=0.0):
        """
        Выполнить один шаг оптимизатора SGD.

        Аргументы:
            params:
                Список TensorRef-параметров.
            lr:
                Learning rate.
            momentum:
                Коэффициент momentum. Если 0.0, используется обычный SGD.
        """
        arr = self._arr_params(params)
        self.lib.tg_sgd_step(arr, len(params), lr, momentum)

    def adam_step(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8, t=1):
        """
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
        """
        arr = self._arr_params(params)
        self.lib.tg_adam_step(arr, len(params), lr, beta1, beta2, eps, t)

    def params_save(self, path, params):
        """
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
        """
        arr = self._arr_params(params)
        st = self.lib.tg_params_save(str(path).encode("utf-8"), arr, len(params))
        if st != 0:
            raise RuntimeError(f"tg_params_save failed with status={st}")

    def params_load(self, path, params):
        """
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
        """
        arr = self._arr_params(params)
        st = self.lib.tg_params_load(str(path).encode("utf-8"), arr, len(params))
        if st != 0:
            raise RuntimeError(f"tg_params_load failed with status={st}")

    def matmul(self, a, b):
        """
        Матричное умножение.

        Формы:
            a: [B x D]
            b: [D x H]

        Возвращает:
            TensorRef формы [B x H].
        """
        h = self.lib.tg_matmul(self.arena, a.handle, b.handle)
        if not h:
            raise RuntimeError("tg_matmul failed")
        return TensorRef(h)

    def add(self, a, b):
        """
        Поэлементное сложение.

        Поддерживаемые формы:
            - одинаковые формы
            - bias-broadcast для случая [R x C] + [1 x C]

        Возвращает:
            TensorRef с результатом.
        """
        h = self.lib.tg_add(self.arena, a.handle, b.handle)
        if not h:
            raise RuntimeError("tg_add failed")
        return TensorRef(h)

    def relu(self, x):
        """
        Применить ReLU-активацию.

        Аргументы:
            x:
                Входной TensorRef.

        Возвращает:
            TensorRef той же формы.
        """
        h = self.lib.tg_relu(self.arena, x.handle)
        if not h:
            raise RuntimeError("tg_relu failed")
        return TensorRef(h)

    def sigmoid(self, x):
        """
        Применить sigmoid-активацию.

        Аргументы:
            x:
                Входной TensorRef.

        Возвращает:
            TensorRef той же формы.
        """
        h = self.lib.tg_sigmoid(self.arena, x.handle)
        if not h:
            raise RuntimeError("tg_sigmoid failed")
        return TensorRef(h)

    def tanh(self, x):
        """
        Применить tanh-активацию.

        Аргументы:
            x:
                Входной TensorRef.

        Возвращает:
            TensorRef той же формы.
        """
        h = self.lib.tg_tanh(self.arena, x.handle)
        if not h:
            raise RuntimeError("tg_tanh failed")
        return TensorRef(h)

    def linear(self, x, weight, bias=None):
        """
        Применить линейный слой.

        Эквивалентно:
            x @ weight + bias

        Формы:
            x:      [B x IN]
            weight: [IN x OUT]
            bias:   None или [1 x OUT]

        Возвращает:
            TensorRef формы [B x OUT].
        """
        bias_h = bias.handle if bias is not None else None
        h = self.lib.tg_linear(self.arena, x.handle, weight.handle, bias_h)
        if not h:
            raise RuntimeError("tg_linear failed")
        return TensorRef(h)

    def mse(self, pred, target):
        """
        Mean squared error loss.

        Формы:
            pred:   [R x C]
            target: [R x C]

        Возвращает:
            Scalar TensorRef формы [1 x 1].
        """
        h = self.lib.tg_mse(self.arena, pred.handle, target.handle)
        if not h:
            raise RuntimeError("tg_mse failed")
        return TensorRef(h)

    def bce_with_logits(self, logits, target):
        """
        Binary cross-entropy loss с logits.

        Формы:
            logits: [R x C]
            target: [R x C]

        Возвращает:
            Scalar TensorRef формы [1 x 1].

        Замечания:
        - Используется численно стабильная fused-реализация BCE-with-logits.
        """
        h = self.lib.tg_bce_with_logits(self.arena, logits.handle, target.handle)
        if not h:
            raise RuntimeError("tg_bce_with_logits failed")
        return TensorRef(h)

    def softmax_cross_entropy(self, logits, target):
        """
        Softmax cross-entropy loss.

        Формы:
            logits: [B x C]
            target: [B x C], one-hot encoded labels

        Возвращает:
            Scalar TensorRef формы [1 x 1].

        Замечания:
        - Используется стабильная fused-реализация softmax + cross-entropy.
        """
        h = self.lib.tg_softmax_cross_entropy(self.arena, logits.handle, target.handle)
        if not h:
            raise RuntimeError("tg_softmax_cross_entropy failed")
        return TensorRef(h)

    def backward(self, loss):
        """
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
        """
        st = self.lib.tg_backward(loss.handle)
        if st != 0:
            raise RuntimeError(f"tg_backward failed with status={st}")
