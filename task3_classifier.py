"""任務 3 (下)：用訓練好的 cGAN 生成器把少數類別補到跟最多的類別一樣多，再訓練同一個 CNN。

- 目標數量與任務 2 相同（happy 的張數），差別只在「補進來的圖」是 GAN 生成而非增強。
- 預設（3a/3b）：缺額全部用 GAN 圖補。
- 改良用法（3c 之後）：`--gan-fraction 0.5` 讓缺額一半用 GAN 圖、一半用任務 2 的影像增強；
  `--filter-model` 用一個已訓練好的分類器篩選 GAN 圖，只保留「被判成指定情緒、且信心 >= --min-conf」的圖。
- 額外檢查：用任務 1 的分類器去辨識生成圖，看生成的「disgust」是否真的像 disgust，
  作為生成品質（條件是否正確）的粗略指標。
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # 必須在 import keras 之前設定

import json

import numpy as np

import keras

from common import CLASSES, MODELS, QUICK, RESULTS, SEED, class_counts, load_data, set_seed, to_model_input, train_and_evaluate
from task2_augment import augment, build_augmenter
from task3_gan import generate


def diversity(x_uint8, n=300, seed=0):
    """隨機取 n 張，計算兩兩之間的平均像素差（0~1）。數值接近 0 代表生成圖幾乎一樣（模式崩塌）。"""
    rng = np.random.default_rng(seed)
    x = x_uint8[rng.choice(len(x_uint8), size=min(n, len(x_uint8)), replace=False)].reshape(-1, 48 * 48) / 255.0
    i, j = np.triu_indices(len(x), k=1)
    return round(float(np.abs(x[i] - x[j]).mean()), 4)


def sample_gan(generator, label, n, filter_model=None, min_conf=0.5, max_rounds=1 if QUICK else 10):
    """產生 n 張指定情緒的 GAN 圖；有 filter_model 時只保留通過篩選的圖。回傳 (影像, 統計)。"""
    if filter_model is None:
        return generate(generator, label, n), {"generated": n, "kept": n}
    kept, generated = [], 0
    for r in range(max_rounds):
        remaining = n - sum(len(k) for k in kept)
        if remaining <= 0:
            break
        cand = generate(generator, label, max(4 * remaining, 512), seed=SEED + 1000 * r)
        probs = filter_model.predict(to_model_input(cand), batch_size=512, verbose=0)
        ok = (probs.argmax(1) == label) & (probs[:, label] >= min_conf)
        generated += len(cand)
        kept.append(cand[ok])
    imgs = np.concatenate(kept)[:n]
    return imgs, {"generated": generated, "kept": int(len(imgs)),
                  "pass_rate": round(sum(len(k) for k in kept) / generated, 4)}


def save_synthetic_examples(x_syn, y_syn, path, per_class=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [c for c in range(len(CLASSES)) if (y_syn == c).any()]
    if not present:  # 例如篩選太嚴、沒有任何 GAN 圖通過
        return
    fig, axes = plt.subplots(len(present), per_class, figsize=(per_class * 0.9, len(present) * 1.0), squeeze=False)
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
    parser.add_argument("--gan-fraction", type=float, default=1.0, help="缺額中用 GAN 圖補的比例，其餘用影像增強")
    parser.add_argument("--filter-model", default=None, help="models/ 底下用來篩選 GAN 圖的分類器檔名")
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="分類器的訓練種子（GAN 圖與增強圖仍用 SEED 產生，只改變模型初始化與訓練順序）")
    args = parser.parse_args()
    OUT = RESULTS / args.task_name
    OUT.mkdir(parents=True, exist_ok=True)
    (xtr, ytr), (xva, yva), (xte, yte) = load_data()
    generator = keras.saving.load_model(MODELS / args.generator)

    filter_model = keras.saving.load_model(MODELS / args.filter_model) if args.filter_model else None
    set_seed()
    rng = np.random.default_rng(SEED)
    augmenter = build_augmenter() if args.gan_fraction < 1 else None

    target = int(np.bincount(ytr).max())
    xs, ys, xa, ya, gan_stats = [], [], [], [], {}
    for c in range(len(CLASSES)):
        need = target - int((ytr == c).sum())
        if need <= 0:
            continue
        n_gan = int(round(need * args.gan_fraction))
        n_kept = 0
        if n_gan > 0:
            imgs, gan_stats[CLASSES[c]] = sample_gan(generator, c, n_gan, filter_model, args.min_conf)
            n_kept = len(imgs)
            xs.append(imgs)
            ys.append(np.full(n_kept, c, dtype=ytr.dtype))
        n_aug = need - n_kept  # GAN 圖不夠時（篩選太嚴）用增強補足
        if n_aug > 0:
            if augmenter is None:
                augmenter = build_augmenter()
            src = rng.choice(np.flatnonzero(ytr == c), size=n_aug, replace=True)
            xa.append(augment(augmenter, xtr[src]))
            ya.append(np.full(n_aug, c, dtype=ytr.dtype))
    x_syn, y_syn = np.concatenate(xs), np.concatenate(ys)
    x_aug = np.concatenate(xa) if xa else np.zeros((0, 48, 48), np.uint8)
    y_aug = np.concatenate(ya) if ya else np.zeros(0, ytr.dtype)
    save_synthetic_examples(x_syn, y_syn, OUT / "synthetic_examples.png")

    # 生成品質檢查：任務 1 的分類器（只看過真實資料）認為生成圖是哪一類？
    fidelity = None
    baseline_path = MODELS / "task1_raw.keras"
    if baseline_path.exists() and len(x_syn) > 0:
        baseline = keras.saving.load_model(baseline_path)
        pred = baseline.predict(to_model_input(x_syn), batch_size=512, verbose=0).argmax(1)
        fidelity = {CLASSES[c]: round(float((pred[y_syn == c] == c).mean()), 4)
                    for c in range(len(CLASSES)) if (y_syn == c).any()}
        fidelity["_note"] = "任務1分類器把該類生成圖判為同一類的比例（真實測試集上的對應數字見 task1 recall）"

    x_mix, y_mix = np.concatenate([xtr, x_syn, x_aug]), np.concatenate([ytr, y_syn, y_aug])
    (OUT / "balance_counts.json").write_text(json.dumps({
        "target_per_class": target, "before": class_counts(ytr), "synthetic_added": class_counts(y_syn), "augmented_added": class_counts(y_aug),
        "gan_fraction": args.gan_fraction, "filter_model": args.filter_model, "min_conf": args.min_conf,
        "gan_sampling": gan_stats,
        "after": class_counts(y_mix), "baseline_agrees_with_gan_label": fidelity,
        "pixel_diversity_synthetic": {CLASSES[c]: diversity(x_syn[y_syn == c]) for c in range(len(CLASSES)) if (y_syn == c).any()},
        "pixel_diversity_real_train": {CLASSES[c]: diversity(xtr[ytr == c]) for c in range(len(CLASSES))},
        "generator": args.generator,
    }, ensure_ascii=False, indent=2))
    train_and_evaluate(args.task_name, x_mix, y_mix, xva, yva, xte, yte, seed=args.seed,
                       notes=(f"[{args.generator}] 真實訓練集 + cGAN 生成圖（缺額的 {args.gan_fraction:.0%}，"
                              f"篩選模型 {args.filter_model}，min_conf {args.min_conf}）+ 影像增強補其餘缺額，"
                              f"每類補到 {target} 張；驗證/測試集只有真實影像。"))
