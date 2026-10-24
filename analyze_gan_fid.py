"""用 FID（Fréchet distance）客觀比較各版 GAN 的生成品質。

標準的 FID 用 ImageNet 訓練的 Inception 網路抽特徵，但它不懂 48x48 灰階的表情。
這裡改用任務 1 的 CNN（只看過真實照片）倒數第二層的 256 維特徵，稱為 FER-FID：
- 對每個類別，比較「生成圖」與「同類別真實訓練圖」的特徵分布（平均與共變異數）。
- 數值越小，生成圖在「表情分類器眼中」越像真實照片。
- 對照組（下限參考）：同類別的真實驗證圖 vs 真實訓練圖。
注意：樣本數少時 FID 會偏高（disgust 只有 371 張真圖、65 張驗證圖），
但所有 GAN 版本用相同的張數計算，所以版本之間的排名仍然有意義。
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # 必須在 import keras 之前設定

import argparse
import json

import numpy as np
from scipy import linalg

import keras

from common import CLASSES, MODELS, RESULTS, load_data, to_model_input
from task3_gan import generate

N_SYN = 2000


def feature_model():
    base = keras.saving.load_model(MODELS / "task1_raw.keras")
    dense = [layer for layer in base.layers if isinstance(layer, keras.layers.Dense)][0]  # Dense(256)
    return keras.Model(base.inputs, dense.output)


def features(model, x_uint8):
    return model.predict(to_model_input(x_uint8), batch_size=512, verbose=0).astype("float64")


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generators", nargs="+",
                        default=["v1=task3_generator.keras", "v2=task3_generator_v2.keras"])
    args = parser.parse_args()

    (xtr, ytr), (xva, yva), _ = load_data()
    fm = feature_model()
    real_train = {c: features(fm, xtr[ytr == c]) for c in range(len(CLASSES))}
    out_path = RESULTS / "task3_gan_fid.json"
    result = json.loads(out_path.read_text()) if out_path.exists() else {}
    result["real_val_reference"] = {
        CLASSES[c]: round(frechet(features(fm, xva[yva == c]), real_train[c]), 2) for c in range(len(CLASSES))}
    for spec in args.generators:
        name, path = spec.split("=")
        gen = keras.saving.load_model(MODELS / path)
        scores = {CLASSES[c]: round(frechet(features(fm, generate(gen, c, N_SYN, seed=777)), real_train[c]), 2)
                  for c in range(len(CLASSES))}
        scores["_mean"] = round(float(np.mean([scores[c] for c in CLASSES])), 2)
        result[name] = scores
        print(name, scores)
    result["real_val_reference"]["_mean"] = round(float(np.mean([result["real_val_reference"][c] for c in CLASSES])), 2)
    result["_note"] = (f"FER-FID：任務1 CNN 的 256 維特徵上的 Fréchet distance，生成圖 {N_SYN} 張 vs 同類別真實訓練圖；"
                       "越小越好。real_val_reference 為真實驗證圖 vs 真實訓練圖的參考值（驗證圖張數較少，數值偏高）。")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("real_val_reference", result["real_val_reference"])
