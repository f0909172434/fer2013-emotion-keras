"""任務 3 (下)：用訓練好的 cGAN 生成器把少數類別補到跟最多的類別一樣多，再訓練同一個 CNN。

- 目標數量與任務 2 相同（happy 的張數），差別只在「補進來的圖」是 GAN 生成而非增強。
- 額外檢查：用任務 1 的分類器去辨識生成圖，看生成的「disgust」是否真的像 disgust，
  作為生成品質（條件是否正確）的粗略指標。
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # 必須在 import keras 之前設定

import json

import numpy as np

import keras

from common import CLASSES, MODELS, RESULTS, class_counts, load_data, to_model_input, train_and_evaluate
from task3_gan import generate


def diversity(x_uint8, n=300, seed=0):
    """隨機取 n 張，計算兩兩之間的平均像素差（0~1）。數值接近 0 代表生成圖幾乎一樣（模式崩塌）。"""
    rng = np.random.default_rng(seed)
    x = x_uint8[rng.choice(len(x_uint8), size=min(n, len(x_uint8)), replace=False)].reshape(-1, 48 * 48) / 255.0
    i, j = np.triu_indices(len(x), k=1)
    return round(float(np.abs(x[i] - x[j]).mean()), 4)


def save_synthetic_examples(x_syn, y_syn, path, per_class=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [c for c in range(len(CLASSES)) if (y_syn == c).any()]
    fig, axes = plt.subplots(len(present), per_class, figsize=(per_class * 0.9, len(present) * 1.0))
    for r, c in enumerate(present):
        idx = np.flatnonzero(y_syn == c)[:per_class]
        for k, i in enumerate(idx):
            axes[r, k].imshow(x_syn[i], cmap="gray", vmin=0, vmax=255)
        for k in range(per_class):
            axes[r, k].axis("off")
        axes[r, 0].set_title(f"{CLASSES[c]} (GAN)", fontsize=7, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", default="task3_generator.keras", help="models/ 底下的生成器檔名")
    parser.add_argument("--task-name", default="task3_gan_classifier_v1")
    args = parser.parse_args()
    OUT = RESULTS / args.task_name
    OUT.mkdir(parents=True, exist_ok=True)
    (xtr, ytr), (xva, yva), (xte, yte) = load_data()
    generator = keras.saving.load_model(MODELS / args.generator)

    target = int(np.bincount(ytr).max())
    xs, ys = [], []
    for c in range(len(CLASSES)):
        need = target - int((ytr == c).sum())
        if need > 0:
            xs.append(generate(generator, c, need))
            ys.append(np.full(need, c, dtype=ytr.dtype))
    x_syn, y_syn = np.concatenate(xs), np.concatenate(ys)
    save_synthetic_examples(x_syn, y_syn, OUT / "synthetic_examples.png")

    # 生成品質檢查：任務 1 的分類器（只看過真實資料）認為生成圖是哪一類？
    fidelity = None
    baseline_path = MODELS / "task1_raw.keras"
    if baseline_path.exists():
        baseline = keras.saving.load_model(baseline_path)
        pred = baseline.predict(to_model_input(x_syn), batch_size=512, verbose=0).argmax(1)
        fidelity = {CLASSES[c]: round(float((pred[y_syn == c] == c).mean()), 4)
                    for c in range(len(CLASSES)) if (y_syn == c).any()}
        fidelity["_note"] = "任務1分類器把該類生成圖判為同一類的比例（真實測試集上的對應數字見 task1 recall）"

    x_mix, y_mix = np.concatenate([xtr, x_syn]), np.concatenate([ytr, y_syn])
    (OUT / "balance_counts.json").write_text(json.dumps({
        "target_per_class": target, "before": class_counts(ytr), "synthetic_added": class_counts(y_syn),
        "after": class_counts(y_mix), "baseline_agrees_with_gan_label": fidelity,
        "pixel_diversity_synthetic": {CLASSES[c]: diversity(x_syn[y_syn == c]) for c in range(len(CLASSES)) if (y_syn == c).any()},
        "pixel_diversity_real_train": {CLASSES[c]: diversity(xtr[ytr == c]) for c in range(len(CLASSES))},
        "generator": args.generator,
    }, ensure_ascii=False, indent=2))
    train_and_evaluate(args.task_name, x_mix, y_mix, xva, yva, xte, yte,
                       notes=f"[{args.generator}] 真實訓練集 + cGAN 生成圖，每類補到 {target} 張；驗證/測試集只有真實影像。")
