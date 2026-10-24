"""任務 1：原始資料不處理，直接訓練情緒分類模型。

「不處理」的定義：只把像素縮放到 [0, 1]（神經網路輸入必要的格式轉換），
不做影像增強、不重新取樣、不加 class weight。
"""
import argparse
import json

from common import RESULTS, SEED, class_counts, load_data, train_and_evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=SEED, help="訓練用種子，用來估計種子造成的波動")
args = parser.parse_args()

(xtr, ytr), (xva, yva), (xte, yte) = load_data()

RESULTS.mkdir(exist_ok=True)
(RESULTS / "data_split_counts.json").write_text(json.dumps({
    "train": class_counts(ytr), "val": class_counts(yva), "test": class_counts(yte),
}, ensure_ascii=False, indent=2))

task = "task1_raw" if args.seed == SEED else f"task1_raw_seed{args.seed}"
train_and_evaluate(task, xtr, ytr, xva, yva, xte, yte, seed=args.seed,
                   notes=f"原始影像，只縮放到 [0,1]；無增強、無平衡。訓練種子 {args.seed}。")
