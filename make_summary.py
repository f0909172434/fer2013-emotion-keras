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
]

metrics = {key: json.loads((RESULTS / key / "metrics.json").read_text()) for key, _ in RUNS}

lines = [
    "| 實驗 | 訓練張數 | 最佳 epoch | 測試準確率 | 測試 macro-F1 | " + " | ".join(f"{c} 召回率" for c in CLASSES) + " |",
    "|" + "---|" * (5 + len(CLASSES)),
]
for key, label in RUNS:
    m = metrics[key]
    recalls = " | ".join(f"{m['test_per_class_recall'][c]:.3f}" for c in CLASSES)
    lines.append(f"| {label} | {m['train_size']} | {m['best_epoch']} | {m['test_accuracy']:.4f} | "
                 f"{m['test_macro_f1']:.4f} | {recalls} |")
(RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(10, 4.2))
width = 0.2
x = np.arange(len(CLASSES))
for i, (key, label) in enumerate(RUNS):
    vals = [metrics[key]["test_per_class_recall"][c] for c in CLASSES]
    ax.bar(x + (i - 1.5) * width, vals, width, label=f"{label} (macro-F1 {metrics[key]['test_macro_f1']:.3f})")
ax.set_xticks(x, CLASSES)
ax.set_ylabel("test recall")
ax.set_ylim(0, 1)
ax.set_title("Per-class test recall (same CNN, same val/test split; only the training set differs)")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(RESULTS / "comparison_recall.png", dpi=110)
plt.close(fig)
