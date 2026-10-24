# 診斷：v3 判別器（sum pooling）的情緒分類頭，在不同 DiffAugment 設定下能否學會分辨情緒（只用真圖，300 步）
import os, sys
os.environ["KERAS_BACKEND"] = "torch"
sys.path.insert(0, os.getcwd())
import numpy as np, torch, keras
import task3_gan_v3 as g
from common import load_data
from task3_gan import balanced_indices

(x, y), _, _ = load_data()
idx = balanced_indices(y, np.random.default_rng(0))[::8]
x = torch.tensor((x[idx].astype("float32") / 127.5 - 1)[..., None], device="mps")
y = torch.tensor(y[idx], device="mps")
ce = keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def no_cutout(t):
    # 與 g.diff_augment 相同，但跳過 cutout：暫時把 cutout 區域設成 0 大小
    n, dev = t.shape[0], t.device
    t = t + (torch.rand(n, 1, 1, 1, device=dev) - 0.5)
    m = t.mean(dim=(1, 2, 3), keepdim=True)
    t = (t - m) * (torch.rand(n, 1, 1, 1, device=dev) + 0.5) + m
    t = t.permute(0, 3, 1, 2)
    theta = torch.zeros(n, 2, 3, device=dev); theta[:, 0, 0] = 1; theta[:, 1, 1] = 1
    theta[:, :, 2] = (torch.rand(n, 2, device=dev) * 2 - 1) * 0.25
    grid = torch.nn.functional.affine_grid(theta, list(t.shape), align_corners=False)
    return torch.nn.functional.grid_sample(t, grid, padding_mode="zeros", align_corners=False).permute(0, 2, 3, 1)


for name, aug in (("full_diffaugment", g.diff_augment), ("no_cutout", no_cutout), ("no_aug", lambda t: t)):
    torch.manual_seed(0)
    D = g.build_discriminator_v3()
    opt = keras.optimizers.Adam(4e-4, beta_1=0.0, beta_2=0.9)
    accs = []
    for step in range(300):
        b = torch.randint(0, len(y), (128,), device="mps")
        xb, yb = x[b], y[b].view(-1, 1).int()
        D.zero_grad()
        _, aux = D([aug(xb), yb], training=True)
        loss = ce(yb.view(-1), aux)
        loss.backward()
        with torch.no_grad():
            opt.apply([v.value.grad for v in D.trainable_weights], D.trainable_weights)
        accs.append(float((aux.argmax(-1) == yb.view(-1)).float().mean()))
    print(f"{name}: aux acc (last 50 steps) = {np.mean(accs[-50:]):.3f}")
