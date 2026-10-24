"""任務 3 (上) 改良版：cGAN v3，針對「小資料集」的已知改良一次加入。

保留 v2 的各類別均衡抽樣，生成器架構與 v1/v2 相同（只改訓練方式與判別器）：
1. DiffAugment：餵給判別器前，對真圖和假圖做同一套「可微分」的隨機增強（平移、挖洞、亮度、對比）。
   判別器只有 371 張 disgust 可看時很容易背圖；增強讓它無法只靠記憶分辨真假，
   而且因為假圖也經過同樣的增強（且可微分），生成器不會學到「把增強效果畫進圖裡」。
2. 判別器加上 Spectral Normalization（限制每層的放大倍率，讓訓練更穩定）並改用 hinge loss。
3. Projection discriminator：用「標籤嵌入 · 影像特徵」的內積表達條件，比 v1/v2 把標籤圖疊在影像上更有效。
4. 輔助分類頭（AC-GAN 的想法）：判別器同時學著分辨情緒，生成器被要求畫出「判得出情緒」的臉。
5. 生成器權重的 EMA（指數移動平均）：取樣用平滑後的權重，畫質通常較穩定。
因為一次改了多項，v2 → v3 的差異歸因於整組改良，而不是單一項目。
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # 必須在 import keras 之前設定

import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import keras
from keras import layers, ops

from common import CLASSES, MODELS, QUICK, RESULTS, SEED, load_data, set_seed
from task3_gan import LATENT_DIM, N_CLASSES, balanced_indices, build_generator, save_grid

GAN_EPOCHS = 1 if QUICK else 200
GAN_TIME_LIMIT_MIN = 150
AUX_WEIGHT = 0.2
EMA_DECAY = 0.999
OUT = RESULTS / "task3_gan_v3"
GEN_PATH = MODELS / "task3_generator_v3.keras"
TAG = "v3"
# v4（--clean-aux）：真假判斷看增強圖，情緒分類頭改看「未增強」的圖。
# 診斷（results/_failed/diag_aux_vs_diffaugment.txt）：DiffAugment 會抹掉表情線索，分類頭學不動。
CLEAN_AUX = False


# ---------- DiffAugment（PyTorch 張量，NHWC，像素範圍 [-1, 1]）----------
def diff_augment(x):
    n = x.shape[0]
    dev = x.device
    # 亮度：整張圖加上 [-0.5, 0.5) 的偏移
    x = x + (torch.rand(n, 1, 1, 1, device=dev) - 0.5)
    # 對比：以每張圖的平均為中心放大/縮小 [0.5, 1.5) 倍
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * (torch.rand(n, 1, 1, 1, device=dev) + 0.5) + mean
    # 平移：上下左右最多 1/8 影像大小，空出來的地方補 0
    x = x.permute(0, 3, 1, 2)  # NCHW
    shift = (torch.rand(n, 2, device=dev) * 2 - 1) * 0.25  # affine_grid 的座標範圍是 [-1, 1]，0.25 = 1/8 圖寬
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    # Cutout：隨機挖掉一塊 24x24（邊長一半）的方塊
    h, w = x.shape[2], x.shape[3]
    cy = torch.randint(0, h, (n, 1, 1), device=dev)
    cx = torch.randint(0, w, (n, 1, 1), device=dev)
    yy = torch.arange(h, device=dev).view(1, h, 1)
    xx = torch.arange(w, device=dev).view(1, 1, w)
    keep = ((yy - cy).abs() >= h // 4) | ((xx - cx).abs() >= w // 4)
    x = x * keep.unsqueeze(1).float()
    return x.permute(0, 2, 3, 1)  # 回到 NHWC


def build_discriminator_v3():
    img = keras.Input((48, 48, 1))
    label = keras.Input((1,), dtype="int32")
    sn = layers.SpectralNormalization
    x = img
    for filters in (64, 128, 256):          # 48 -> 24 -> 12 -> 6
        x = sn(layers.Conv2D(filters, 4, strides=2, padding="same"))(x)
        x = layers.LeakyReLU(0.2)(x)
    x = sn(layers.Conv2D(256, 3, padding="same"))(x)
    x = layers.LeakyReLU(0.2)(x)
    # 全域「加總」池化（SN-GAN 的標準做法）：SN 讓每層都不放大訊號，若再取平均，特徵會小到判別器幾乎學不動
    # （診斷實驗：平均池化時分類頭 300 步只到 20.5%，改加總後 31.2%，與不加 SN 的 31.8% 相當）
    h = layers.Rescaling(6.0 * 6.0)(layers.GlobalAveragePooling2D()(x))  # 256 維影像特徵
    # projection：真假分數 = w·h + embed(label)·h
    uncond = sn(layers.Dense(1))(h)
    emb = layers.Flatten()(layers.Embedding(N_CLASSES, 256)(label))
    proj = layers.Dot(axes=1)([emb, h])
    logit = layers.Add()([uncond, proj])
    aux = layers.Dense(N_CLASSES)(h)         # 輔助分類頭：輸出 7 類的 logits
    return keras.Model([img, label], [logit, aux], name="discriminator_v3")


class ConditionalGANv3(keras.Model):
    def __init__(self, generator, discriminator, ema_generator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.ema_generator = ema_generator
        self.seed_gen = keras.random.SeedGenerator(SEED)
        self.ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.step_count = 0
        self.trackers = {name: keras.metrics.Mean(name=name) for name in
                         ("d_loss", "g_loss", "d_real_acc", "d_fake_acc", "aux_real_acc", "aux_fake_acc")}

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def call(self, inputs):
        # fit() 會先呼叫一次模型來完成 build；GAN 的實際運算都在 train_step 裡
        return inputs

    @property
    def metrics(self):
        return list(self.trackers.values())

    def _sample(self, n):
        z = keras.random.normal((n, LATENT_DIM), seed=self.seed_gen)
        lab = ops.cast(keras.random.randint((n, 1), 0, N_CLASSES, seed=self.seed_gen), "int32")
        return z, lab

    def _update_ema(self):
        self.step_count += 1
        decay = min(EMA_DECAY, (1 + self.step_count) / (10 + self.step_count))  # 前期用較小的 decay 暖身
        with torch.no_grad():
            # weights 包含 BatchNorm 的 moving mean/variance，一起平滑
            for ema_v, v in zip(self.ema_generator.weights, self.generator.weights):
                ema_v.value.mul_(decay).add_(v.value, alpha=1 - decay)

    def train_step(self, data):
        real, labels = data
        labels = ops.cast(ops.reshape(labels, (-1, 1)), "int32")
        n = ops.shape(real)[0]

        # (1) 訓練 D：hinge loss + 對真圖做情緒分類
        z, fake_lab = self._sample(n)
        fake = self.generator([z, fake_lab], training=True).detach()
        self.zero_grad()
        # 真圖與假圖合成一個 batch 一次送進 D：Spectral Normalization 在每次 forward 時會就地更新權重，
        # 分兩次呼叫會讓第一次的計算圖失效
        imgs = [diff_augment(ops.concatenate([real, fake], axis=0))]
        labs = [labels, fake_lab]
        if CLEAN_AUX:  # 多附上一份未增強的真圖，只給情緒分類頭用
            imgs.append(real)
            labs.append(labels)
        logit, aux = self.discriminator([ops.concatenate(imgs, axis=0), ops.concatenate(labs, axis=0)], training=True)
        real_logit, fake_logit = logit[:n], logit[n:2 * n]
        real_aux = aux[2 * n:] if CLEAN_AUX else aux[:n]
        d_adv = ops.mean(ops.relu(1.0 - real_logit)) + ops.mean(ops.relu(1.0 + fake_logit))
        d_loss = d_adv + AUX_WEIGHT * self.ce(ops.reshape(labels, (-1,)), real_aux)
        d_loss.backward()
        d_vars = self.discriminator.trainable_weights
        with torch.no_grad():
            self.d_optimizer.apply([v.value.grad for v in d_vars], d_vars)

        # (2) 訓練 G：讓 D 給高分，並讓輔助分類頭認得出指定的情緒
        z, fake_lab = self._sample(n)
        self.zero_grad()
        gen = self.generator([z, fake_lab], training=True)
        if CLEAN_AUX:
            logit, aux = self.discriminator(
                [ops.concatenate([diff_augment(gen), gen], axis=0), ops.concatenate([fake_lab, fake_lab], axis=0)],
                training=True)
            gen_logit, gen_aux = logit[:n], aux[n:]
        else:
            gen_logit, gen_aux = self.discriminator([diff_augment(gen), fake_lab], training=True)
        g_loss = -ops.mean(gen_logit) + AUX_WEIGHT * self.ce(ops.reshape(fake_lab, (-1,)), gen_aux)
        g_loss.backward()
        g_vars = self.generator.trainable_weights
        with torch.no_grad():
            self.g_optimizer.apply([v.value.grad for v in g_vars], g_vars)
        self._update_ema()

        t = self.trackers
        t["d_loss"].update_state(d_loss)
        t["g_loss"].update_state(g_loss)
        t["d_real_acc"].update_state(ops.mean(ops.cast(real_logit > 0, "float32")))
        t["d_fake_acc"].update_state(ops.mean(ops.cast(fake_logit < 0, "float32")))
        t["aux_real_acc"].update_state(ops.mean(ops.cast(ops.argmax(real_aux, -1) == ops.reshape(labels, (-1,)), "float32")))
        t["aux_fake_acc"].update_state(ops.mean(ops.cast(ops.argmax(gen_aux, -1) == ops.reshape(fake_lab, (-1,)), "float32")))
        return {k: m.result() for k, m in t.items()}


class GanMonitorV3(keras.callbacks.Callback):
    def __init__(self, every=5):
        super().__init__()
        self.every = every
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every == 0 or epoch == 0:
            save_grid(self.model.ema_generator, OUT / f"samples_epoch{epoch + 1:03d}.png",
                      f"cGAN {TAG} (EMA) samples, epoch {epoch + 1}")
        if time.time() - self.t0 > GAN_TIME_LIMIT_MIN * 60:
            print(f"time limit reached after epoch {epoch + 1}")
            self.model.stop_training = True


def plot_v3_history(h, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(h["d_loss"], label="D loss (hinge + aux)")
    axes[0].plot(h["g_loss"], label="G loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    for key in ("d_real_acc", "d_fake_acc", "aux_real_acc", "aux_fake_acc"):
        axes[1].plot(h[key], label=key)
    axes[1].set_xlabel("epoch")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def build_gan():
    generator = build_generator()
    ema_generator = build_generator()
    ema_generator.set_weights(generator.get_weights())
    gan = ConditionalGANv3(generator, build_discriminator_v3(), ema_generator)
    # TTUR：判別器學習率較大（SAGAN/BigGAN 的常見設定）
    gan.compile(
        d_optimizer=keras.optimizers.Adam(4e-4, beta_1=0.0, beta_2=0.9),
        g_optimizer=keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.9),
    )
    return gan


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-aux", action="store_true", help="v4：情緒分類頭看未增強的圖（其他與 v3 相同）")
    parser.add_argument("--time-limit-min", type=float, default=GAN_TIME_LIMIT_MIN)
    parser.add_argument("--aux-weight", type=float, default=AUX_WEIGHT, help="輔助分類損失的權重")
    parser.add_argument("--tag", default=None, help="版本名稱，決定結果與模型的檔名（例如 v5）")
    args = parser.parse_args()
    CLEAN_AUX, AUX_WEIGHT, GAN_TIME_LIMIT_MIN = args.clean_aux, args.aux_weight, args.time_limit_min
    TAG = args.tag or ("v4" if args.clean_aux else "v3")
    OUT, GEN_PATH = RESULTS / f"task3_gan_{TAG}", MODELS / f"task3_generator_{TAG}.keras"

    set_seed()
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    (xtr, ytr), _, _ = load_data()
    idx = balanced_indices(ytr, np.random.default_rng(SEED))  # 與 v2 相同的均衡抽樣
    xtr, ytr = xtr[idx], ytr[idx]
    x = (xtr.astype("float32") / 127.5 - 1.0)[..., None]

    gan = build_gan()
    t0 = time.time()
    hist = gan.fit(x, ytr, batch_size=128, epochs=GAN_EPOCHS, shuffle=True, verbose=2,
                   callbacks=[GanMonitorV3(every=5), keras.callbacks.CSVLogger(str(OUT / "gan_history.csv"))])
    gan.ema_generator.save(GEN_PATH)
    epochs_run = len(hist.history["d_loss"])
    save_grid(gan.ema_generator, OUT / "samples_final.png", f"cGAN {TAG} (EMA) samples, final (epoch {epochs_run})")
    plot_v3_history(hist.history, OUT / "gan_losses.png")
    (OUT / "gan_summary.json").write_text(json.dumps({
        "version": TAG, "clean_aux": CLEAN_AUX, "aux_weight": AUX_WEIGHT, "time_limit_min": GAN_TIME_LIMIT_MIN, "balanced_sampling": True, "train_images_per_epoch": int(len(ytr)),
        "epochs_run": epochs_run, "steps_run": gan.step_count, "train_seconds": round(time.time() - t0, 1),
        "latent_dim": LATENT_DIM, "batch_size": 128,
        "changes_vs_v2": ["DiffAugment(brightness, contrast, translation 1/8, cutout 1/2) on real+fake for D and G steps",
                          "spectral normalization in D", "hinge loss", "projection discriminator",
                          f"auxiliary classifier head (weight {AUX_WEIGHT})", f"generator EMA (decay {EMA_DECAY})",
                          "TTUR Adam: D 4e-4, G 1e-4, betas (0.0, 0.9)"],
        "final": {k: float(v[-1]) for k, v in hist.history.items()},
    }, indent=2))
