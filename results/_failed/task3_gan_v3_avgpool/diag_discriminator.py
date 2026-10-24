import os, sys
os.environ["KERAS_BACKEND"] = "torch"
sys.path.insert(0, os.getcwd())
import numpy as np, torch, keras
from keras import layers, ops
from common import load_data
from task3_gan import balanced_indices
(x, y), _, _ = load_data()
idx = balanced_indices(y, np.random.default_rng(0))[::8]
x = torch.tensor((x[idx].astype("float32") / 127.5 - 1)[..., None], device="mps"); y = torch.tensor(y[idx], device="mps")
ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def build(sn_on, pool):
    img = keras.Input((48, 48, 1)); h = img
    wrap = layers.SpectralNormalization if sn_on else (lambda l: l)
    for f in (64, 128, 256):
        h = layers.LeakyReLU(0.2)(wrap(layers.Conv2D(f, 4, strides=2, padding="same"))(h))
    h = layers.LeakyReLU(0.2)(wrap(layers.Conv2D(256, 3, padding="same"))(h))
    h = layers.GlobalAveragePooling2D()(h)
    if pool == "sum":
        h = layers.Rescaling(36.0)(h)
    return keras.Model(img, layers.Dense(7)(h))

for sn_on, pool in ((False, "avg"), (True, "avg"), (True, "sum")):
    D = build(sn_on, pool); opt = keras.optimizers.Adam(4e-4, beta_1=0.0, beta_2=0.9); accs = []
    for step in range(300):
        b = torch.randint(0, len(y), (128,), device="mps")
        D.zero_grad(); aux = D(x[b], training=True); loss = ce(y[b], aux); loss.backward()
        with torch.no_grad(): opt.apply([v.value.grad for v in D.trainable_weights], D.trainable_weights)
        accs.append(float((aux.argmax(-1) == y[b]).float().mean()))
    print(f"sn={sn_on} pool={pool} acc_last50={np.mean(accs[-50:]):.3f}")
