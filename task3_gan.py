"""任務 3 (上)：訓練條件式 DCGAN (cGAN)，可指定要生成哪一種情緒的 48x48 灰階人臉。

- 生成器 G：輸入隨機雜訊 z 與情緒標籤，輸出一張假臉。
- 判別器 D：輸入一張臉與它的標籤，判斷是真的還是 G 做的。
- 兩者輪流訓練：D 學著分辨真假，G 學著騙過 D。
只用訓練集（不含驗證集/測試集）訓練。
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # 必須在 import keras 之前設定

import json
import time

import numpy as np
import torch

import keras
from keras import layers, ops

from common import CLASSES, MODELS, RESULTS, SEED, load_data, set_seed

LATENT_DIM = 128
GAN_EPOCHS = 100          # 上限
GAN_TIME_LIMIT_MIN = 70   # 牆鐘時間上限，兩者先到者停止
N_CLASSES = len(CLASSES)


def build_generator():
    z = keras.Input((LATENT_DIM,))
    label = keras.Input((1,), dtype="int32")
    emb = layers.Flatten()(layers.Embedding(N_CLASSES, 50)(label))
    x = layers.Concatenate()([z, emb])
    x = layers.Dense(6 * 6 * 256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((6, 6, 256))(x)
    for filters in (128, 64, 32):           # 6 -> 12 -> 24 -> 48
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
    img = layers.Conv2D(1, 3, padding="same", activation="tanh")(x)  # 輸出範圍 [-1, 1]
    return keras.Model([z, label], img, name="generator")


def build_discriminator():
    img = keras.Input((48, 48, 1))
    label = keras.Input((1,), dtype="int32")
    # 把標籤變成一張 48x48 的「標籤圖」，和影像疊在一起，讓 D 判斷「這張臉像不像這個情緒的真臉」
    lab_map = layers.Reshape((48, 48, 1))(layers.Dense(48 * 48)(layers.Flatten()(layers.Embedding(N_CLASSES, 50)(label))))
    x = layers.Concatenate()([img, lab_map])
    for filters in (64, 128, 256):          # 48 -> 24 -> 12 -> 6
        x = layers.Conv2D(filters, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    logit = layers.Dense(1)(x)              # 輸出 logit，>0 代表「真」
    return keras.Model([img, label], logit, name="discriminator")


class ConditionalGAN(keras.Model):
    """依 Keras 官方「用 PyTorch 後端自訂 train_step」的寫法實作 GAN 的交替訓練。"""

    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.seed_gen = keras.random.SeedGenerator(SEED)
        self.bce = keras.losses.BinaryCrossentropy(from_logits=True)
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.d_real_acc = keras.metrics.Mean(name="d_real_acc")
        self.d_fake_acc = keras.metrics.Mean(name="d_fake_acc")

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def call(self, inputs):
        # fit() 會先呼叫一次模型來完成 build；GAN 的實際運算都在 train_step 裡
        return inputs

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker, self.d_real_acc, self.d_fake_acc]

    def _sample(self, n):
        z = keras.random.normal((n, LATENT_DIM), seed=self.seed_gen)
        lab = ops.cast(keras.random.randint((n, 1), 0, N_CLASSES, seed=self.seed_gen), "int32")
        return z, lab

    def train_step(self, data):
        real, labels = data
        labels = ops.cast(ops.reshape(labels, (-1, 1)), "int32")
        n = ops.shape(real)[0]

        # (1) 訓練 D：真圖標 0.9（label smoothing，避免 D 過度自信），假圖標 0
        z, fake_lab = self._sample(n)
        fake = self.generator([z, fake_lab], training=True).detach()
        self.zero_grad()
        real_logit = self.discriminator([real, labels], training=True)
        fake_logit = self.discriminator([fake, fake_lab], training=True)
        d_loss = self.bce(ops.ones_like(real_logit) * 0.9, real_logit) + self.bce(ops.zeros_like(fake_logit), fake_logit)
        d_loss.backward()
        d_vars = self.discriminator.trainable_weights
        with torch.no_grad():
            self.d_optimizer.apply([v.value.grad for v in d_vars], d_vars)

        # (2) 訓練 G：希望 D 把假圖判成真
        z, fake_lab = self._sample(n)
        self.zero_grad()
        gen_logit = self.discriminator([self.generator([z, fake_lab], training=True), fake_lab], training=True)
        g_loss = self.bce(ops.ones_like(gen_logit), gen_logit)
        g_loss.backward()
        g_vars = self.generator.trainable_weights
        with torch.no_grad():
            self.g_optimizer.apply([v.value.grad for v in g_vars], g_vars)

        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        self.d_real_acc.update_state(ops.mean(ops.cast(real_logit > 0, "float32")))
        self.d_fake_acc.update_state(ops.mean(ops.cast(fake_logit < 0, "float32")))
        return {m.name: m.result() for m in self.metrics}


def generate(generator, label, n, seed=SEED, batch=512):
    """產生 n 張指定情緒的假臉，回傳 uint8 (n, 48, 48)。"""
    rng = np.random.default_rng(seed + label)
    out = []
    for i in range(0, n, batch):
        m = min(batch, n - i)
        z = rng.standard_normal((m, LATENT_DIM)).astype("float32")
        lab = np.full((m, 1), label, dtype="int32")
        img = keras.ops.convert_to_numpy(generator([z, lab], training=False))
        out.append(((img[..., 0] + 1.0) * 127.5).clip(0, 255).round().astype(np.uint8))
    return np.concatenate(out)


def save_grid(generator, path, title, per_class=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(N_CLASSES, per_class, figsize=(per_class * 0.9, N_CLASSES * 1.0))
    for r in range(N_CLASSES):
        imgs = generate(generator, r, per_class, seed=1234)  # 固定雜訊，方便比較不同 epoch
        for k in range(per_class):
            axes[r, k].imshow(imgs[k], cmap="gray", vmin=0, vmax=255)
            axes[r, k].axis("off")
        axes[r, 0].set_title(CLASSES[r], fontsize=7, loc="left")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


class GanMonitor(keras.callbacks.Callback):
    def __init__(self, out, every=10):
        super().__init__()
        self.out = out
        self.every = every
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every == 0 or epoch == 0:
            save_grid(self.model.generator, self.out / f"samples_epoch{epoch + 1:03d}.png", f"cGAN samples, epoch {epoch + 1}")
        if time.time() - self.t0 > GAN_TIME_LIMIT_MIN * 60:
            print(f"time limit reached after epoch {epoch + 1}")
            self.model.stop_training = True


def plot_gan_history(h, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(h["d_loss"], label="D loss")
    axes[0].plot(h["g_loss"], label="G loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(h["d_real_acc"], label="D acc on real")
    axes[1].plot(h["d_fake_acc"], label="D acc on fake")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def balanced_indices(y, rng):
    """每個類別都抽到和最多類別一樣多的索引（少數類別重複抽樣）。"""
    target = int(np.bincount(y).max())
    idx = [rng.choice(np.flatnonzero(y == c), size=target, replace=True) for c in range(N_CLASSES)]
    return np.concatenate(idx)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced", action="store_true",
                        help="v2：每個類別均衡抽樣真實影像（其他設定與 v1 完全相同）")
    args = parser.parse_args()
    out = RESULTS / ("task3_gan_v2" if args.balanced else "task3_gan")
    gen_path = MODELS / ("task3_generator_v2.keras" if args.balanced else "task3_generator.keras")

    set_seed()
    out.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    (xtr, ytr), _, _ = load_data()
    if args.balanced:
        # v1 的問題：一個 batch 平均只有約 2 張真的 disgust，卻有約 18 張標成 disgust 的假圖，
        # D 很快學到「disgust 標籤 = 假」，G 對 disgust 拿不到有用的梯度 → 模式崩塌。
        idx = balanced_indices(ytr, np.random.default_rng(SEED))
        xtr, ytr = xtr[idx], ytr[idx]
    x = (xtr.astype("float32") / 127.5 - 1.0)[..., None]  # GAN 慣例：縮放到 [-1, 1] 配合 tanh

    gan = ConditionalGAN(build_generator(), build_discriminator())
    gan.compile(
        d_optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
        g_optimizer=keras.optimizers.Adam(2e-4, beta_1=0.5),
    )
    t0 = time.time()
    hist = gan.fit(x, ytr, batch_size=128, epochs=GAN_EPOCHS, shuffle=True, verbose=2,
                   callbacks=[GanMonitor(out, every=10), keras.callbacks.CSVLogger(str(out / "gan_history.csv"))])
    gan.generator.save(gen_path)
    epochs_run = len(hist.history["d_loss"])
    save_grid(gan.generator, out / "samples_final.png", f"cGAN samples, final (epoch {epochs_run})")
    plot_gan_history(hist.history, out / "gan_losses.png")
    (out / "gan_summary.json").write_text(json.dumps({
        "balanced_sampling": args.balanced, "train_images_per_epoch": int(len(ytr)),
        "epochs_run": epochs_run, "steps_run": epochs_run * int(np.ceil(len(ytr) / 128)),
        "train_seconds": round(time.time() - t0, 1),
        "latent_dim": LATENT_DIM, "batch_size": 128, "optimizer": "Adam(2e-4, beta_1=0.5) for G and D",
        "real_label_smoothing": 0.9, "final": {k: float(v[-1]) for k, v in hist.history.items()},
    }, indent=2))
