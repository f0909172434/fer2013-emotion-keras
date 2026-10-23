"""任務 1：原始資料不處理，直接訓練情緒分類模型。

「不處理」的定義：只把像素縮放到 [0, 1]（神經網路輸入必要的格式轉換），
不做影像增強、不重新取樣、不加 class weight。
"""
import json

from common import RESULTS, class_counts, load_data, train_and_evaluate

(xtr, ytr), (xva, yva), (xte, yte) = load_data()

RESULTS.mkdir(exist_ok=True)
(RESULTS / "data_split_counts.json").write_text(json.dumps({
    "train": class_counts(ytr), "val": class_counts(yva), "test": class_counts(yte),
}, ensure_ascii=False, indent=2))

train_and_evaluate("task1_raw", xtr, ytr, xva, yva, xte, yte,
                   notes="原始影像，只縮放到 [0,1]；無增強、無平衡。")
