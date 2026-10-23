"""檢查 GAN 是否只是「背下」訓練圖：比較生成圖與真實圖到最近訓練圖的距離。

- 對每個類別，取 200 張生成圖，計算它與同類別「最像的一張訓練圖」的平均像素差。
- 對照組：驗證集的真實圖（模型沒看過的新照片）到最近訓練圖的距離。
- 生成圖的距離若遠小於對照組，代表生成圖很接近某張訓練圖（記憶/複製）；
  接近對照組代表它和「一張新照片」一樣新。
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # 必須在 import keras 之前設定

import json

import numpy as np

import keras

from common import CLASSES, MODELS, RESULTS, load_data
from task3_gan import generate

N = 200


def nearest_distance(queries, refs, chunk=50):
    q = queries.reshape(len(queries), -1).astype("float32") / 255.0
    r = refs.reshape(len(refs), -1).astype("float32") / 255.0
    out = []
    for i in range(0, len(q), chunk):
        d = np.abs(q[i:i + chunk, None, :] - r[None, :, :]).mean(axis=2)
        out.append(d.min(axis=1))
    return np.concatenate(out)


if __name__ == "__main__":
    (xtr, ytr), (xva, yva), _ = load_data()
    rng = np.random.default_rng(0)
    result = {}
    for version, path in (("v1", "task3_generator.keras"), ("v2", "task3_generator_v2.keras")):
        gen = keras.saving.load_model(MODELS / path)
        result[version] = {}
        for c, name in enumerate(CLASSES):
            refs = xtr[ytr == c]
            syn = generate(gen, c, N, seed=999)
            result[version][name] = round(float(nearest_distance(syn, refs).mean()), 4)
    result["real_val_reference"] = {}
    for c, name in enumerate(CLASSES):
        val = xva[yva == c]
        val = val[rng.choice(len(val), size=min(N, len(val)), replace=False)]
        result["real_val_reference"][name] = round(float(nearest_distance(val, xtr[ytr == c]).mean()), 4)
    result["_note"] = "平均『到最近同類訓練圖』的像素差（0~1）。越小越像某張訓練圖。"
    (RESULTS / "task3_gan_memorization.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
