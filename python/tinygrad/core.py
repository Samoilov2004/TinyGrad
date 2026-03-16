import ctypes
import os
from pathlib import Path


class TensorRef:
    def __init__(self, handle, backing=None, rows=None, cols=None):
        self.handle = handle
        self.backing = backing
        self.rows = rows
        self.cols = cols


def _find_library():
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
    def __init__(self, lib_path=None, arena_bytes=8 * 1024 * 1024):
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
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        if self.arena:
            self.lib.tg_arena_destroy(self.arena)
            self.arena = None

    def reset(self):
        self.lib.tg_arena_reset(self.arena)

    def param(self, rows, cols, requires_grad=True):
        h = self.lib.tg_param_create(rows, cols, requires_grad)
        if not h:
            raise RuntimeError("tg_param_create failed")
        return TensorRef(h, None, rows, cols)

    def free_param(self, tensor):
        if tensor and tensor.handle:
            self.lib.tg_param_free(tensor.handle)
            tensor.handle = None
            tensor.backing = None

    def tensor_from_rows(self, rows_2d, requires_grad=False):
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
        return int(self.lib.tg_numel(tensor.handle))

    def data_1d(self, tensor):
        n = self.numel(tensor)
        ptr = self.lib.tg_tensor_data(tensor.handle)
        return [float(ptr[i]) for i in range(n)]

    def data_2d(self, tensor, rows=None, cols=None):
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
        ptr = self.lib.tg_tensor_data(tensor.handle)
        return float(ptr[0])

    def copy_into_param(self, tensor, flat_values):
        n = self.numel(tensor)
        if len(flat_values) != n:
            raise ValueError("size mismatch")
        ptr = self.lib.tg_tensor_data(tensor.handle)
        for i, v in enumerate(flat_values):
            ptr[i] = float(v)

    def zero_grads(self, params):
        for p in params:
            self.lib.tg_zero_grad(p.handle)

    def _arr_params(self, params):
        return (self.c_void_p * len(params))(*[p.handle for p in params])

    def sgd_step(self, params, lr, momentum=0.0):
        arr = self._arr_params(params)
        self.lib.tg_sgd_step(arr, len(params), lr, momentum)

    def adam_step(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8, t=1):
        arr = self._arr_params(params)
        self.lib.tg_adam_step(arr, len(params), lr, beta1, beta2, eps, t)

    def params_save(self, path, params):
        arr = self._arr_params(params)
        st = self.lib.tg_params_save(str(path).encode("utf-8"), arr, len(params))
        if st != 0:
            raise RuntimeError(f"tg_params_save failed with status={st}")

    def params_load(self, path, params):
        arr = self._arr_params(params)
        st = self.lib.tg_params_load(str(path).encode("utf-8"), arr, len(params))
        if st != 0:
            raise RuntimeError(f"tg_params_load failed with status={st}")

    # ops
    def matmul(self, a, b):
        h = self.lib.tg_matmul(self.arena, a.handle, b.handle)
        if not h:
            raise RuntimeError("tg_matmul failed")
        return TensorRef(h)

    def add(self, a, b):
        h = self.lib.tg_add(self.arena, a.handle, b.handle)
        if not h:
            raise RuntimeError("tg_add failed")
        return TensorRef(h)

    def relu(self, x):
        h = self.lib.tg_relu(self.arena, x.handle)
        if not h:
            raise RuntimeError("tg_relu failed")
        return TensorRef(h)

    def sigmoid(self, x):
        h = self.lib.tg_sigmoid(self.arena, x.handle)
        if not h:
            raise RuntimeError("tg_sigmoid failed")
        return TensorRef(h)

    def tanh(self, x):
        h = self.lib.tg_tanh(self.arena, x.handle)
        if not h:
            raise RuntimeError("tg_tanh failed")
        return TensorRef(h)

    def linear(self, x, weight, bias=None):
        bias_h = bias.handle if bias is not None else None
        h = self.lib.tg_linear(self.arena, x.handle, weight.handle, bias_h)
        if not h:
            raise RuntimeError("tg_linear failed")
        return TensorRef(h)

    # losses
    def mse(self, pred, target):
        h = self.lib.tg_mse(self.arena, pred.handle, target.handle)
        if not h:
            raise RuntimeError("tg_mse failed")
        return TensorRef(h)

    def bce_with_logits(self, logits, target):
        h = self.lib.tg_bce_with_logits(self.arena, logits.handle, target.handle)
        if not h:
            raise RuntimeError("tg_bce_with_logits failed")
        return TensorRef(h)

    def softmax_cross_entropy(self, logits, target):
        h = self.lib.tg_softmax_cross_entropy(self.arena, logits.handle, target.handle)
        if not h:
            raise RuntimeError("tg_softmax_cross_entropy failed")
        return TensorRef(h)

    def backward(self, loss):
        st = self.lib.tg_backward(loss.handle)
        if st != 0:
            raise RuntimeError(f"tg_backward failed with status={st}")
