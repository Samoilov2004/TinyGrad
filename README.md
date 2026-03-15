# TinyGrad

Сейчас в проекте есть:
- базовый 2D tensor `float32`
- `grad` и `requires_grad`
- arena allocator для временных объектов
- динамический graph / op-node
- `tg_backward()` для scalar loss
- базовые операции:
  - `tg_add`
  - `tg_sub`
  - `tg_mul`
  - `tg_matmul`
  - `tg_sum`
  - `tg_mean`

## Структура

- `include/tg.h` — главный внешний API
- `src/tensor.c` — tensor API
- `src/arena.c` — arena allocator
- `src/autograd.c` — topo sort + backward
- `src/ops_basic.c` — базовые операции и их backward
- `src/tg.c` — version string

## Структура
- `include/tg.h` - главный внешний API

## Сборка
#### Дебаг
```Bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

#### Release
```Bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Запуск
#### Примеры
```Bash
./build/xor
./build/logreg_csv path/to/data.csv
```

#### Тесты
```Bash
./build/gradcheck
ctest --test-dir build --output-on-failure
```
