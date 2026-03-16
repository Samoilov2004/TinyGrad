import inspect
import sys
from pathlib import Path


def repo_root():
    return Path(__file__).resolve().parent.parent


def prepare_import_path():
    root = repo_root()
    python_dir = root / "python"
    sys.path.insert(0, str(python_dir))


def clean_doc(doc):
    if not doc:
        return "Документация отсутствует."
    return inspect.cleandoc(doc)


def md_escape(text):
    return text.replace("_", "\\_")


def slugify(title):
    """
    Простейший slug для markdown-оглавления.
    Достаточно хороший для локальной документации.
    """
    s = title.strip().lower()
    s = s.replace("`", "")
    s = s.replace(".", "")
    s = s.replace("(", "")
    s = s.replace(")", "")
    s = s.replace(",", "")
    s = s.replace("/", "")
    s = s.replace(":", "")
    s = s.replace("[", "")
    s = s.replace("]", "")
    s = s.replace(" ", "-")
    return s


def format_signature(obj):
    try:
        return str(inspect.signature(obj))
    except Exception:
        return "(...)"


def public_methods(cls):
    items = []
    for name, obj in inspect.getmembers(cls):
        if inspect.isfunction(obj):
            items.append((name, obj))
    return items


def split_summary_and_body(doc):
    """
    Разделяет docstring на:
    - краткое первое предложение / первую строку
    - остальной текст
    """
    doc = clean_doc(doc)
    parts = doc.split("\n\n", 1)
    summary = parts[0].strip()
    body = parts[1].strip() if len(parts) > 1 else ""
    return summary, body


def render_doc_block(doc):
    doc = clean_doc(doc)
    return doc


def render_method(name, fn, level=4):
    sig = format_signature(fn)
    title = f"`{name}{sig}`"
    header = "#" * level + f" {title}"
    body = render_doc_block(inspect.getdoc(fn))
    return [header, "", body, ""]


def render_constructor(cls, level=4):
    init_fn = cls.__init__
    sig = format_signature(init_fn)
    title = f"`{cls.__name__}{sig.replace('(self, ', '(').replace('(self)', '()')}`"
    header = "#" * level + f" {title}"
    body = render_doc_block(inspect.getdoc(init_fn))
    return [header, "", body, ""]


def method_map(cls):
    return {name: fn for name, fn in public_methods(cls)}


def render_class_header(cls):
    lines = []
    lines.append(f"## `{cls.__name__}`")
    lines.append("")
    lines.append(render_doc_block(inspect.getdoc(cls)))
    lines.append("")
    return lines


def render_tensorref_section(TensorRef):
    lines = []
    lines.extend(render_class_header(TensorRef))
    lines.extend(render_constructor(TensorRef, level=3))
    return lines


def render_tinygradc_section(TinyGradC):
    lines = []
    methods = method_map(TinyGradC)

    groups = [
        (
            "Runtime",
            ["close", "reset"],
        ),
        (
            "Создание и lifetime tensor",
            ["param", "free_param", "tensor_from_rows"],
        ),
        (
            "Чтение и работа с данными",
            ["numel", "data_1d", "data_2d", "scalar", "copy_into_param", "zero_grads"],
        ),
        (
            "Операции",
            ["matmul", "add", "relu", "sigmoid", "tanh", "linear"],
        ),
        (
            "Функции потерь",
            ["mse", "bce_with_logits", "softmax_cross_entropy", "backward"],
        ),
        (
            "Оптимизаторы",
            ["sgd_step", "adam_step"],
        ),
        (
            "Сохранение и загрузка параметров",
            ["params_save", "params_load"],
        ),
    ]

    lines.extend(render_class_header(TinyGradC))
    lines.extend(render_constructor(TinyGradC, level=3))

    for group_title, names in groups:
        present = [(name, methods[name]) for name in names if name in methods]
        if not present:
            continue

        lines.append(f"### {group_title}")
        lines.append("")

        for name, fn in present:
            lines.extend(render_method(f"TinyGradC.{name}", fn, level=4))

    return lines


def render_toc():
    sections = [
        ("Обзор", "обзор"),
        ("TensorRef", "`tensorref`"),
        ("TinyGradC", "`tinygradc`"),
        ("TinyGradC / Runtime", "runtime"),
        ("TinyGradC / Создание и lifetime tensor", "создание-и-lifetime-tensor"),
        ("TinyGradC / Чтение и работа с данными", "чтение-и-работа-с-данными"),
        ("TinyGradC / Операции", "операции"),
        ("TinyGradC / Функции потерь", "функции-потерь"),
        ("TinyGradC / Оптимизаторы", "оптимизаторы"),
        ("TinyGradC / Сохранение и загрузка параметров", "сохранение-и-загрузка-параметров"),
    ]

    lines = []
    lines.append("## Оглавление")
    lines.append("")
    lines.append("- [Обзор](#обзор)")
    lines.append("- [`TensorRef`](#tensorref)")
    lines.append("- [`TinyGradC`](#tinygradc)")
    lines.append("  - [Runtime](#runtime)")
    lines.append("  - [Создание и lifetime tensor](#создание-и-lifetime-tensor)")
    lines.append("  - [Чтение и работа с данными](#чтение-и-работа-с-данными)")
    lines.append("  - [Операции](#операции)")
    lines.append("  - [Функции потерь](#функции-потерь)")
    lines.append("  - [Оптимизаторы](#оптимизаторы)")
    lines.append("  - [Сохранение и загрузка параметров](#сохранение-и-загрузка-параметров)")
    lines.append("")
    return lines


def render_overview():
    lines = []
    lines.append("## Обзор")
    lines.append("")
    lines.append(
        "Этот файл автоматически генерируется из docstring'ов в "
        "`python/tinygradc/core.py`."
    )
    lines.append("")
    lines.append("Основные классы Python API:")
    lines.append("")
    lines.append("- `TensorRef` — лёгкая ссылка на tensor в C runtime")
    lines.append("- `TinyGradC` — основной Python runtime wrapper")
    lines.append("")
    lines.append("Типичный сценарий использования:")
    lines.append("")
    lines.append("```python")
    lines.append("from tinygradc import TinyGradC")
    lines.append("")
    lines.append("with TinyGradC() as tg:")
    lines.append("    w = tg.param(64, 10, True)")
    lines.append("    b = tg.param(1, 10, True)")
    lines.append("")
    lines.append("    x = tg.tensor_from_rows(batch_x, requires_grad=False)")
    lines.append("    y = tg.tensor_from_rows(batch_y, requires_grad=False)")
    lines.append("")
    lines.append("    logits = tg.linear(x, w, b)")
    lines.append("    loss = tg.softmax_cross_entropy(logits, y)")
    lines.append("")
    lines.append("    tg.backward(loss)")
    lines.append("    tg.adam_step([w, b], lr=1e-3, t=1)")
    lines.append("    tg.reset()")
    lines.append("```")
    lines.append("")
    return lines


def main():
    prepare_import_path()

    from tinygradc.core import TensorRef, TinyGradC

    lines = []
    lines.append("# Python API: tinygradc")
    lines.append("")
    lines.append("> Этот файл сгенерирован автоматически. Не редактируй его вручную.")
    lines.append("")

    lines.extend(render_toc())
    lines.extend(render_overview())
    lines.extend(render_tensorref_section(TensorRef))
    lines.append("")
    lines.extend(render_tinygradc_section(TinyGradC))
    lines.append("")

    docs_dir = repo_root() / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    out_path = docs_dir / "python_api.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
