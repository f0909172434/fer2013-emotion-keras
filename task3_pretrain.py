"""任務 3 的另一種用法：GAN 圖只用來「預訓練」CNN，再用任務 2 的資料（真圖 + 增強圖）微調。

前面的實驗顯示，把 GAN 圖直接混進訓練集，GAN 圖比例越高 macro-F1 越低
（0% → 0.631、50% 篩選後 → 0.597、100% → 0.586）。這裡改成兩階段：
1. 預訓練：只用 GAN v2 生成的均衡資料（每類 6,133 張，共 42,931 張）訓練同一個 CNN，
   以真實驗證集提早停止。這個模型在真實測試集上的表現就是 CAS（Classification Accuracy Score），
   衡量「只靠 GAN 圖能學到多少真實表情」，是評估條件式 GAN 的常用指標。
2. 微調：以預訓練權重為起點，用任務 2 的資料與完全相同的訓練設定訓練，和任務 2 直接比較。
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # 必須在 import keras 之前設定

import argparse
import json

import numpy as np

import keras

from common import (BATCH_SIZE, CLASSES, MODELS, QUICK, RESULTS, SEED, build_cnn, class_counts, load_data, set_seed,
                    to_model_input, train_and_evaluate)
from task2_augment import balance_with_augmentation
from task3_gan import generate

PRETRAIN_MAX_EPOCHS = 1 if QUICK else 20

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", default="task3_generator_v2.keras")
    parser.add_argument("--task-name", default="task3f_v2_pretrain_finetune")
    args = parser.parse_args()
    out = RESULTS / args.task_name
    out.mkdir(parents=True, exist_ok=True)

    (xtr, ytr), (xva, yva), (xte, yte) = load_data()
    generator = keras.saving.load_model(MODELS / args.generator)
    target = int(np.bincount(ytr).max())
    x_gan = np.concatenate([generate(generator, c, target, seed=SEED + 50) for c in range(len(CLASSES))])
    y_gan = np.repeat(np.arange(len(CLASSES)), target).astype(ytr.dtype)

    # 1. 只用 GAN 圖預訓練
    set_seed()
    model = build_cnn()
    hist = model.fit(
        to_model_input(x_gan), y_gan, validation_data=(to_model_input(xva), yva),
        batch_size=BATCH_SIZE, epochs=PRETRAIN_MAX_EPOCHS, shuffle=True, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
    from sklearn.metrics import f1_score
    pred = model.predict(to_model_input(xte), batch_size=512, verbose=0).argmax(1)
    pretrain = {
        "generator": args.generator, "gan_train_counts": class_counts(y_gan),
        "epochs_run": len(hist.history["loss"]), "best_val_accuracy": float(max(hist.history["val_accuracy"])),
        "cas_test_accuracy": float((pred == yte).mean()), "cas_test_macro_f1": float(f1_score(yte, pred, average="macro")),
        "_note": "CAS：只用 GAN 圖訓練的分類器在真實測試集上的表現。對照：任務 1（真圖）macro-F1 0.603。",
    }
    (out / "pretrain_gan_only.json").write_text(json.dumps(pretrain, ensure_ascii=False, indent=2))
    print(json.dumps(pretrain, ensure_ascii=False))

    # 2. 用任務 2 的資料微調（與任務 2 完全相同的資料、設定與種子，只差起始權重）
    x_bal, y_bal, _ = balance_with_augmentation(xtr, ytr)
    train_and_evaluate(args.task_name, x_bal, y_bal, xva, yva, xte, yte, initial_weights=model.get_weights(),
                       notes=f"先用 {args.generator} 生成的均衡資料預訓練，再以任務 2 的資料與設定微調。")
