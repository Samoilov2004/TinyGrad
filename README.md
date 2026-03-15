# TinyGrad

Проект по самописному пакету на С11, нормальная дока позже, ридми версии 2

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
