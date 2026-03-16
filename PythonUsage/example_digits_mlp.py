import csv
import math
import random
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt

import tinygradc


# Скачивание данных
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

TRAIN_FILE = DATA_DIR / "optdigits.tra"
TEST_FILE = DATA_DIR / "optdigits.tes"

TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra"
TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes"


def download_if_missing(path, url):
    if path.exists():
        print(f"found: {path}")
        return
    print(f"downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)


download_if_missing(TRAIN_FILE, TRAIN_URL)
download_if_missing(TEST_FILE, TEST_URL)


# Загрузка датасета
def load_optdigits_csv(path):
    X = []
    y = []

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            vals = [int(v) for v in row]
            features = vals[:-1]
            label = vals[-1]

            features = [v / 16.0 for v in features]

            X.append(features)
            y.append(label)

    return X, y


X_train, y_train = load_optdigits_csv(TRAIN_FILE)
X_test, y_test = load_optdigits_csv(TEST_FILE)

print("train size:", len(X_train))
print("test size :", len(X_test))
print("feature dim:", len(X_train[0]))
print("first label:", y_train[0])


# Вспомогательные функции, в идеале потом перепишем на си, чтобы было в едином формате
NUM_FEATURES = 64
NUM_CLASSES = 10

def one_hot(labels, num_classes=10):
    out = []
    for y in labels:
        row = [0.0] * num_classes
        row[y] = 1.0
        out.append(row)
    return out

def batch_iter(X, y, batch_size, shuffle=True):
    indices = list(range(len(X)))
    if shuffle:
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        xb = [X[i] for i in batch_idx]
        yb = [y[i] for i in batch_idx]
        yield xb, yb

def argmax(row):
    best_i = 0
    best_v = row[0]
    for i, v in enumerate(row):
        if v > best_v:
            best_v = v
            best_i = i
    return best_i

def accuracy_from_logits(logits_2d, labels):
    correct = 0
    for logit_row, y in zip(logits_2d, labels):
        if argmax(logit_row) == y:
            correct += 1
    return correct / len(labels)

def image8x8_from_flat(flat64):
    img = []
    k = 0
    for _ in range(8):
        row = []
        for _ in range(8):
            row.append(flat64[k])
            k += 1
        img.append(row)
    return img

def xavier_uniform_values(rows, cols, rng):
    limit = math.sqrt(6.0 / (rows + cols))
    vals = []
    for _ in range(rows * cols):
        vals.append(rng.uniform(-limit, limit))
    return vals

def zeros(n):
    return [0.0] * n


# Посмотрим че там
fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for i, ax in enumerate(axes.flat):
    img = image8x8_from_flat(X_train[i])
    ax.imshow(img, cmap="gray_r", vmin=0.0, vmax=1.0)
    ax.set_title(f"label={y_train[i]}")
    ax.axis("off")

plt.tight_layout()
plt.show()
