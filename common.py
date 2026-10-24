"""三個任務共用的程式：讀資料、切驗證集、建立 CNN、訓練與評估。

所有任務使用同一個 CNN、同一組超參數、同一個驗證集與測試集，
唯一改變的是「訓練集」的內容，這樣三個結果才能公平比較。
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # Keras 3 以 PyTorch 為後端，可用 Apple GPU (MPS)

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import keras
from keras import layers

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "extracted"
CACHE = ROOT / "data" / "fer2013_uint8.npz"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SEED = 42
VAL_FRACTION = 0.15

# 固定訓練設定（三個任務完全相同）
BATCH_SIZE = 128
MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 8

torch.set_num_threads(4)  # 機器上還有其他工作在跑，限制 CPU 執行緒


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def _load_split(split):
    xs, ys = [], []
    for label, name in enumerate(CLASSES):
        for path in sorted((DATA_DIR / split / name).glob("*.jpg")):
            img = Image.open(path).convert("L")
            assert img.size == (48, 48), path
            xs.append(np.asarray(img, dtype=np.uint8))
            ys.append(label)
    return np.stack(xs), np.array(ys, dtype=np.int64)


def load_data():
    """回傳 (x_train, y_train), (x_val, y_val), (x_test, y_test)，影像為 uint8 (N, 48, 48)。

    原始資料集只有 train/ 與 test/，所以從 train/ 依類別比例分層抽出 15% 當驗證集。
    驗證集只用來挑 epoch，測試集只在最後評估一次。
    """
    if CACHE.exists():
        d = np.load(CACHE)
        return (d["xtr"], d["ytr"]), (d["xva"], d["yva"]), (d["xte"], d["yte"])

    x_all, y_all = _load_split("train")
    x_test, y_test = _load_split("test")

    rng = np.random.default_rng(SEED)
    val_idx = []
    for c in range(len(CLASSES)):
        idx = np.flatnonzero(y_all == c)
        rng.shuffle(idx)
        val_idx.extend(idx[: int(round(len(idx) * VAL_FRACTION))])
    val_mask = np.zeros(len(y_all), dtype=bool)
    val_mask[val_idx] = True

    xtr, ytr = x_all[~val_mask], y_all[~val_mask]
    xva, yva = x_all[val_mask], y_all[val_mask]
    np.savez_compressed(CACHE, xtr=xtr, ytr=ytr, xva=xva, yva=yva, xte=x_test, yte=y_test)
    return (xtr, ytr), (xva, yva), (x_test, y_test)


def class_counts(y):
    return {name: int((y == i).sum()) for i, name in enumerate(CLASSES)}


def to_model_input(x_uint8):
    """唯一的「處理」：轉成 float32、縮放到 [0, 1]、加上通道維度。"""
    return (x_uint8.astype("float32") / 255.0)[..., None]


def build_cnn():
    """VGG 風格的小型 CNN：4 個卷積區塊 + 全連接分類層。"""
    inputs = keras.Input(shape=(48, 48, 1))
    x = inputs
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="emotion_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(task_name, x_train, y_train, x_val, y_val, x_test, y_test, notes=None, seed=SEED,
                       initial_weights=None):
    """用固定設定訓練 CNN，並把指標、混淆矩陣、學習曲線存到 results/<task_name>/。"""
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    set_seed(seed)  # 只影響權重初始化、dropout 與打亂順序；資料切分固定
    out = RESULTS / task_name
    out.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    # 先打亂一次，讓增強/生成的影像不會集中在尾端；fit() 之後每個 epoch 也會再打亂
    perm = np.random.default_rng(seed).permutation(len(y_train))
    x_train, y_train = x_train[perm], y_train[perm]

    model = build_cnn()
    if initial_weights is not None:  # 例如先用 GAN 圖預訓練過的權重；優化器與訓練設定仍完全相同
        model.set_weights(initial_weights)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        keras.callbacks.CSVLogger(str(out / "history.csv")),
    ]
    t0 = time.time()
    history = model.fit(
        to_model_input(x_train), y_train,
        validation_data=(to_model_input(x_val), y_val),
        batch_size=BATCH_SIZE, epochs=MAX_EPOCHS, callbacks=callbacks, verbose=2,
    )
    train_seconds = time.time() - t0
    model.save(MODELS / f"{task_name}.keras")

    probs = model.predict(to_model_input(x_test), batch_size=512, verbose=0)
    pred = probs.argmax(axis=1)
    report = classification_report(y_test, pred, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=list(range(len(CLASSES))))

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    metrics = {
        "task": task_name,
        "notes": notes,
        "train_counts": class_counts(y_train),
        "train_size": int(len(y_train)),
        "epochs_run": len(history.history["loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": float(min(history.history["val_loss"])),
        "best_val_accuracy": float(history.history["val_accuracy"][best_epoch - 1]),
        "test_accuracy": float(report["accuracy"]),
        "test_macro_f1": float(f1_score(y_test, pred, average="macro")),
        "test_per_class_recall": {c: round(report[c]["recall"], 4) for c in CLASSES},
        "test_per_class_f1": {c: round(report[c]["f1-score"], 4) for c in CLASSES},
        "confusion_matrix": cm.tolist(),
        "train_seconds": round(train_seconds, 1),
        "config": {
            "seed": seed, "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "lr": LEARNING_RATE,
            "early_stop_patience": EARLY_STOP_PATIENCE, "keras": keras.__version__,
            "backend": keras.backend.backend(), "torch": torch.__version__,
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    plot_history(history.history, out / "learning_curve.png", task_name)
    plot_confusion(cm, out / "confusion_matrix.png", task_name)
    print(json.dumps({k: metrics[k] for k in ("task", "test_accuracy", "test_macro_f1", "best_epoch")}))
    return metrics


def plot_history(h, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, ("loss", "accuracy")):
        ax.plot(h[key], label="train")
        ax.plot(h[f"val_{key}"], label="val")
        ax.set_title(f"{title}: {key}")
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_confusion(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title} (row-normalized recall)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
