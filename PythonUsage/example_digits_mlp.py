import csv
import math
import random
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt

from tinygradc import TinyGradC


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


# Инициализация модели
tg = TinyGradC()

HIDDEN = 32

W1 = tg.param(NUM_FEATURES, HIDDEN, True)
b1 = tg.param(1, HIDDEN, True)
W2 = tg.param(HIDDEN, NUM_CLASSES, True)
b2 = tg.param(1, NUM_CLASSES, True)

params = [W1, b1, W2, b2]

rng = random.Random(42)
tg.copy_into_param(W1, xavier_uniform_values(NUM_FEATURES, HIDDEN, rng))
tg.copy_into_param(b1, zeros(HIDDEN))
tg.copy_into_param(W2, xavier_uniform_values(HIDDEN, NUM_CLASSES, rng))
tg.copy_into_param(b2, zeros(NUM_CLASSES))

print("model initialized: 64 -> 32 -> 10")


# Forward & predict
def forward_mlp(tg, x_t, W1, b1, W2, b2):
    h = tg.linear(x_t, W1, b1)
    h = tg.relu(h)
    logits = tg.linear(h, W2, b2)
    return logits


def predict_logits_dataset(tg, X, batch_size=128):
    all_logits = []

    dummy_labels = [0] * len(X)
    for xb, _ in batch_iter(X, dummy_labels, batch_size, shuffle=False):
        x_t = tg.tensor_from_rows(xb, requires_grad=False)
        logits_t = forward_mlp(tg, x_t, W1, b1, W2, b2)
        logits = tg.data_2d(logits_t, rows=len(xb), cols=NUM_CLASSES)
        all_logits.extend(logits)
        tg.reset()

    return all_logits


def predict_labels_dataset(tg, X, batch_size=128):
    logits = predict_logits_dataset(tg, X, batch_size=batch_size)
    return [argmax(row) for row in logits]


# Main train Loop
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-2

history_loss = []
history_train_acc = []
history_test_acc = []

step = 0

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    num_batches = 0

    for xb, yb_labels in batch_iter(X_train, y_train, BATCH_SIZE, shuffle=True):
        yb = one_hot(yb_labels, NUM_CLASSES)

        tg.zero_grads(params)

        x_t = tg.tensor_from_rows(xb, requires_grad=False)
        y_t = tg.tensor_from_rows(yb, requires_grad=False)

        logits_t = forward_mlp(tg, x_t, W1, b1, W2, b2)
        loss_t = tg.softmax_cross_entropy(logits_t, y_t)

        loss_value = tg.scalar(loss_t)
        tg.backward(loss_t)

        step += 1
        tg.adam_step(params, lr=LR, beta1=0.9, beta2=0.999, eps=1e-8, t=step)

        total_loss += loss_value
        num_batches += 1

        tg.reset()

    train_logits = predict_logits_dataset(tg, X_train, batch_size=256)
    test_logits = predict_logits_dataset(tg, X_test, batch_size=256)

    avg_loss = total_loss / max(1, num_batches)
    train_acc = accuracy_from_logits(train_logits, y_train)
    test_acc = accuracy_from_logits(test_logits, y_test)

    history_loss.append(avg_loss)
    history_train_acc.append(train_acc)
    history_test_acc.append(test_acc)

    print(
        f"epoch={epoch:02d} "
        f"loss={avg_loss:.4f} "
        f"train_acc={train_acc:.4f} "
        f"test_acc={test_acc:.4f}"
    )


# Графики обучения
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(1, len(history_loss) + 1), history_loss, marker="o")
axes[0].set_title("Training loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True)

axes[1].plot(range(1, len(history_train_acc) + 1), history_train_acc, marker="o", label="train")
axes[1].plot(range(1, len(history_test_acc) + 1), history_test_acc, marker="o", label="test")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Сохранение весов
while True:
    save_weight_input = input("Сохранить веса? Y/n: ")
    if save_weight_input.lower() == 'y':
        save_weight_bool = True
        break
    elif save_weight_input.lower() == 'n':
        save_weight_bool = True
        break
    else:
        print("Неправильный формат ввода.")

if save_weight_bool:
    MODEL_PATH = "example_digits_mlp.bin"
    tg.params_save(MODEL_PATH, params)
    print("saved:", MODEL_PATH)
