"""彙整所有任務的測試集結果：輸出比較表 (results/summary.md) 與各類別召回率比較圖。"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import CLASSES, RESULTS

RUNS = [
    ("task1_raw", "1. raw"),
    ("task2_augment", "2. augment-balanced"),
    ("task3_gan_classifier_v1", "3a. cGAN v1 (collapsed)"),
    ("task3_gan_classifier_v2", "3b. cGAN v2 (balanced GAN)"),
    ("task3c_v2_mix_filter", "3c. v2 filtered + augment"),
    ("task3e_v5_mix_filter", "3e. v5 filtered + augment"),
    ("task3f_v2_pretrain_finetune", "3f. GAN pretrain + task-2 finetune"),
]
# 換訓練種子重跑的對照，用來估計種子造成的波動
SEED_RUNS = [("task1_raw_seed1", "1. raw (seed 1)"), ("task2_augment_seed1", "2. augment-balanced (seed 1)"),
             ("task3e_v5_mix_filter_seed1", "3e. v5 filtered + augment (seed 1)")]
SEED_RUNS = [(k, label) for k, label in SEED_RUNS if (RESULTS / k / "metrics.json").exists()]

RUNS = [(k, label) for k, label in RUNS if (RESULTS / k / "metrics.json").exists()]
metrics = {key: json.loads((RESULTS / key / "metrics.json").read_text()) for key, _ in RUNS + SEED_RUNS}

lines = [
    "| 實驗 | 訓練張數 | 最佳 epoch | 測試準確率 | 測試 macro-F1 | " + " | ".join(f"{c} 召回率" for c in CLASSES) + " |",
    "|" + "---|" * (5 + len(CLASSES)),
]
for key, label in RUNS + SEED_RUNS:
    m = metrics[key]
    recalls = " | ".join(f"{m['test_per_class_recall'][c]:.3f}" for c in CLASSES)
    lines.append(f"| {label} | {m['train_size']} | {m['best_epoch']} | {m['test_accuracy']:.4f} | "
                 f"{m['test_macro_f1']:.4f} | {recalls} |")
(RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(11, 4.4))
width = 0.8 / len(RUNS)
x = np.arange(len(CLASSES))
for i, (key, label) in enumerate(RUNS):
    vals = [metrics[key]["test_per_class_recall"][c] for c in CLASSES]
    ax.bar(x + (i - (len(RUNS) - 1) / 2) * width, vals, width, label=f"{label} (macro-F1 {metrics[key]['test_macro_f1']:.3f})")
ax.set_xticks(x, CLASSES)
ax.set_ylabel("test recall")
ax.set_ylim(0, 1)
ax.set_title("Per-class test recall (same CNN, same val/test split; only the training set differs)")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(RESULTS / "comparison_recall.png", dpi=110)
plt.close(fig)
