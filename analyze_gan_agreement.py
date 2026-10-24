"""標籤一致率：用任務 1 的分類器（只看過真實照片）辨識未經篩選的生成圖，判成「指定情緒」的比例。

每個版本、每個類別各生成 1,000 張（固定雜訊種子），比較條件式 GAN 是否真的畫出了被要求的表情。
注意：這個指標和 FER-FID 都使用任務 1 的 CNN，生成器若剛好產生這個 CNN 偏好的特徵，兩者可能一起被「騙」。
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # 必須在 import keras 之前設定

import argparse
import json

import numpy as np

import keras

from common import CLASSES, MODELS, RESULTS, to_model_input
from task3_classifier import diversity
from task3_gan import generate

N = 1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generators", nargs="+",
                        default=["v1=task3_generator.keras", "v2=task3_generator_v2.keras",
                                 "v3=task3_generator_v3.keras", "v5=task3_generator_v5.keras"])
    args = parser.parse_args()
    baseline = keras.saving.load_model(MODELS / "task1_raw.keras")
    result = {}
    for name, path in (spec.split("=") for spec in args.generators):
        gen = keras.saving.load_model(MODELS / path)
        agree, div = {}, {}
        for c, cls in enumerate(CLASSES):
            x = generate(gen, c, N, seed=555)
            pred = baseline.predict(to_model_input(x), batch_size=512, verbose=0).argmax(1)
            agree[cls] = round(float((pred == c).mean()), 3)
            div[cls] = diversity(x)
        agree["_mean"] = round(float(np.mean([agree[c] for c in CLASSES])), 3)
        div["_mean"] = round(float(np.mean([div[c] for c in CLASSES])), 4)
        result[name] = {"label_agreement": agree, "pixel_diversity": div}
        print(name, agree)
    result["_note"] = (f"每類 {N} 張未篩選生成圖；label_agreement = 任務1分類器判成指定情緒的比例；"
                       "pixel_diversity = 隨機 300 張兩兩平均像素差（真實訓練圖約 0.27）。")
    (RESULTS / "task3_gan_label_agreement.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
