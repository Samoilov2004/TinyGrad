# TinyGrad
`tinygradc` — это маленькая библиотека на C для работы с тензорами, вычисления градиентов и обучения простых нейросетей. Проект может быть интересен тем, кому интересно, как условный PyTorch работает под капотом, так что излишние оптимизации были опущены.

Сейчас уже реализовано:
- тензоры `float32`
- автоматическое вычисление градиентов
- набор базовых операций
- функции потерь
- оптимизаторы
- примеры обучения

## Как попробовать?
Проект подготовлен под использование из под питона, скачать можно так:
```Bash
python3 -m venv tinygradvenv
source tinygradvenv/bin/activate

pip install git+https://github.com/Samoilov2004/TinyGrad.git
```

Для API также написана [документация из docstrings](docs/python_api.md).

В папке `PythonUsage` лежат примеры использования из под питона.
- `example_digits_mlp.py` - обучение простой сети для угадывания цифр

## Структура кода
- `include/tg.h` - главный внешний API
- `src/tg.c` - служебные штуки, например версия проекта
- `src/tensor.c` - создание, удаление, размер, доступ к данным, работа с grad для тензоров
- `src/arena.c` - временная память для промежуточных объектов во время вычислений
- `src/autograd.c` - код для backward и работы вычислительного графа
- `src/ops_basic.c` - Базовые математические операции:
  - add
  - sub
  - mul
  - matmul
  - sum
  - mean
- `src/nn_layer.c` - миниамальный nn-слой
- `src/data.c` - DataLoader
- `src/train_utils.c` - L2 и clipping
- `src/ops_nn.c` - реализации `relu`, `sigmoid`, `tanh`
- `src/ops_loss.c` - реализация `SGD`, `Adam`, `zeroing gradients`, `parameter list`
- `src/io.c` - работа с весами модели
- `examples/` - примеры использования
- `tests/gradcheck.c` - тесты для градиентов

## Техническая информация
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
./build/xor_mlp
./build/logreg_csv path/to/data.csv
```

#### Тесты
```Bash
./build/gradcheck
ctest --test-dir build --output-on-failure
```

## Что предстоит сделать
### Удобство архитектурное
- [ ] Регуляризация и стабилизация обучения
- [ ] Сериализация

### Архитектурные
- [ ] Ускорение matmul через CBLAS
- [ ] Многопоточность для операций
