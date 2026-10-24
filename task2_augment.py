"""任務 2：用影像增強把少數類別補到跟最多的類別一樣多，再訓練同一個 CNN。

做法（離線過採樣）：
- 目標數量 = 訓練集中最多的類別（happy）的張數。
- 原始影像全部保留、不改動；不足的張數從該類別原圖隨機挑選，經隨機增強後產生新圖。
- 驗證集與測試集完全不動，所以和任務 1 的比較是公平的。
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # 必須在 import keras 之前設定

import json

import numpy as np

import keras
from keras import layers

from common import CLASSES, RESULTS, SEED, class_counts, load_data, set_seed, train_and_evaluate


def build_augmenter():
    # 適合人臉表情的溫和增強：左右翻轉不改變表情；旋轉/平移/縮放模擬拍攝角度與裁切差異
    return keras.Sequential([
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.04, fill_mode="nearest", seed=SEED),        # 約 ±15 度
        layers.RandomTranslation(0.08, 0.08, fill_mode="nearest", seed=SEED),
        layers.RandomZoom((-0.1, 0.1), fill_mode="nearest", seed=SEED),
        layers.RandomContrast(0.2, seed=SEED),
        layers.RandomBrightness(0.15, value_range=(0, 1), seed=SEED),
    ], name="augmenter")


def augment(augmenter, x_uint8, batch=512):
    out = []
    for i in range(0, len(x_uint8), batch):
        xb = (x_uint8[i:i + batch].astype("float32") / 255.0)[..., None]
        yb = keras.ops.convert_to_numpy(augmenter(xb, training=True))
        out.append(np.clip(yb[..., 0] * 255.0, 0, 255).round().astype(np.uint8))
    return np.concatenate(out)


def balance_with_augmentation(xtr, ytr):
    set_seed()
    rng = np.random.default_rng(SEED)
    augmenter = build_augmenter()
    target = int(np.bincount(ytr).max())
    xs, ys = [xtr], [ytr]
    for c in range(len(CLASSES)):
        idx = np.flatnonzero(ytr == c)
        need = target - len(idx)
        if need <= 0:
            continue
        src = rng.choice(idx, size=need, replace=True)
        xs.append(augment(augmenter, xtr[src]))
        ys.append(np.full(need, c, dtype=ytr.dtype))
    return np.concatenate(xs), np.concatenate(ys), target


def save_examples(xtr, ytr, x_bal, y_bal, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_orig = len(ytr)
    fig, axes = plt.subplots(len(CLASSES), 8, figsize=(8, 7.5))
    rng = np.random.default_rng(0)
    for r, name in enumerate(CLASSES):
        orig = rng.choice(np.flatnonzero(ytr == r), 4, replace=False)
        new_idx = np.flatnonzero(y_bal[n_orig:] == r) + n_orig
        new = rng.choice(new_idx, 4, replace=False) if len(new_idx) >= 4 else []
        for k, i in enumerate(list(orig) + list(new)):
            axes[r, k].imshow(x_bal[i], cmap="gray", vmin=0, vmax=255)
        for k in range(8):
            axes[r, k].axis("off")
        axes[r, 0].set_title(name, fontsize=8, loc="left")
    fig.suptitle("left 4: original | right 4: augmented (happy has none)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED,
                        help="訓練用種子（增強資料固定用 SEED 產生，只改變模型初始化與訓練順序），用來估計種子造成的波動")
    args = parser.parse_args()
    task = "task2_augment" if args.seed == SEED else f"task2_augment_seed{args.seed}"

    (xtr, ytr), (xva, yva), (xte, yte) = load_data()
    x_bal, y_bal, target = balance_with_augmentation(xtr, ytr)
    out = RESULTS / task
    out.mkdir(parents=True, exist_ok=True)
    (out / "balance_counts.json").write_text(json.dumps({
        "target_per_class": target, "before": class_counts(ytr), "after": class_counts(y_bal),
    }, ensure_ascii=False, indent=2))
    if args.seed == SEED:
        save_examples(xtr, ytr, x_bal, y_bal, out / "augmented_examples.png")
    train_and_evaluate(task, x_bal, y_bal, xva, yva, xte, yte, seed=args.seed,
                       notes=f"影像增強離線過採樣，每類補到 {target} 張；驗證/測試集不動。訓練種子 {args.seed}。")
