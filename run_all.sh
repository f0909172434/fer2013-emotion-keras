#!/usr/bin/env bash
# 依序執行本作業的所有實驗（順序有相依性：後面的步驟會用到前面訓練出的模型）。
#
#   bash run_all.sh                  完整重現（Apple M5 約 15 小時，有 NVIDIA GPU 會快很多）
#   FER_QUICK=1 bash run_all.sh      快速模式：每個訓練只跑 1 個 epoch，約 30 分鐘確認流程能跑通，
#                                    輸出寫到 results_quick/、models_quick/，不覆蓋正式結果
#
# 可用環境變數 PYTHON 指定 Python（預設 .venv/bin/python）。
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-.venv/bin/python}"
export KERAS_BACKEND=torch

if [ ! -d data/extracted/train ] || [ ! -d data/extracted/test ]; then
  echo "找不到 data/extracted/train 與 data/extracted/test，請先依 README 第 1.3 節下載 FER2013。" >&2
  exit 1
fi

step() { echo; echo "==== $* ===="; }

step "任務 1：原始資料"
"$PY" task1_baseline.py
step "任務 2：影像增強平衡"
"$PY" task2_augment.py

step "任務 3 第一輪：GAN v1、v2 與 3a、3b"
"$PY" task3_gan.py
"$PY" task3_gan.py --balanced
"$PY" task3_classifier.py --generator task3_generator.keras --task-name task3_gan_classifier_v1
"$PY" task3_classifier.py --generator task3_generator_v2.keras --task-name task3_gan_classifier_v2

step "任務 3 第二輪：改用法（3c、3f）"
"$PY" task3_classifier.py --generator task3_generator_v2.keras --task-name task3c_v2_mix_filter \
  --gan-fraction 0.5 --filter-model task2_augment.keras
"$PY" task3_pretrain.py

step "任務 3 第二輪：改品質（GAN v3、v5 與 3e）"
"$PY" task3_gan_v3.py
"$PY" task3_gan_v3.py --clean-aux --aux-weight 1.0 --tag v5 --time-limit-min 90
"$PY" task3_classifier.py --generator task3_generator_v5.keras --task-name task3e_v5_mix_filter \
  --gan-fraction 0.5 --filter-model task2_augment.keras

step "換種子重跑（估計波動）"
"$PY" task1_baseline.py --seed 1
"$PY" task2_augment.py --seed 1
"$PY" task3_classifier.py --generator task3_generator_v5.keras --task-name task3e_v5_mix_filter_seed1 \
  --gan-fraction 0.5 --filter-model task2_augment.keras --seed 1

step "GAN 品質檢查與彙整"
GENS="v1=task3_generator.keras v2=task3_generator_v2.keras v3=task3_generator_v3.keras v5=task3_generator_v5.keras"
"$PY" analyze_gan_fid.py --generators $GENS
"$PY" analyze_gan_memorization.py --generators $GENS
"$PY" analyze_gan_agreement.py --generators $GENS
"$PY" make_summary.py

if [ "${FER_QUICK:-}" = "1" ]; then OUT=results_quick/; else OUT=results/; fi
echo; echo "完成。結果在 $OUT（總表：${OUT}summary.md）"
