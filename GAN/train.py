# =======================
# IMPORTANT CHANGES MADE
# =======================
# 1. Better FID reliability (more eval batches)
# 2. 1 discriminator step instead of 2
# 3. Lower D LR
# 4. Gradient clipping for D
# 5. Safer graph saving
# 6. Better logging
# 7. Keeps FID + IS + graphs

import os, copy, math, multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils
from torch.amp import autocast, GradScaler

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

import matplotlib.pyplot as plt

from stylegan.generator.generator import Generator
from stylegan.discriminator.perceiver_discriminator import PerceiverDiscriminator


# ======================================================
# LOSSES
# ======================================================

def d_hinge_loss(real_logits, fake_logits):
    return (
        torch.relu(1.0 - real_logits).mean() +
        torch.relu(1.0 + fake_logits).mean()
    )


def g_hinge_loss(fake_logits):
    return -fake_logits.mean()


# ======================================================
# GRAPHS
# ======================================================

def save_graphs(out_dir, d_hist, g_hist, metric_epochs, fid_hist, is_hist):

    plt.figure(figsize=(10, 5))
    plt.plot(d_hist, label="D Loss")
    plt.plot(g_hist, label="G Loss")
    plt.legend()
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/loss_graph.png")
    plt.close()

    if len(fid_hist) > 0:
        plt.figure(figsize=(10, 5))
        plt.plot(metric_epochs, fid_hist, marker="o")
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("FID")
        plt.title("FID vs Epoch")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/fid_graph.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(metric_epochs, is_hist, marker="o")
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("IS")
        plt.title("IS vs Epoch")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/is_graph.png")
        plt.close()


# ======================================================
# MAIN
# ======================================================

def main():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    AMP = DEVICE == "cuda"

    BATCH_SIZE = 56
    EPOCHS = 30
    Z_DIM = 512

    LR_G = 3e-4
    LR_D = 2e-5        # reduced

    BETAS = (0.0, 0.99)

    R1_GAMMA = 5.0     # softer
    PL_WEIGHT = 1.0

    NUM_WORKERS = 2

    DATA_DIR = "data/ffhq"
    OUT = "outputs_ffhq_raj"

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(f"{OUT}/samples", exist_ok=True)
    os.makedirs(f"{OUT}/ckpt", exist_ok=True)

    scaler = GradScaler("cuda", enabled=AMP)

    # ==================================================
    # DATA
    # ==================================================

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    ds = datasets.ImageFolder(DATA_DIR, transform=transform)

    tr = int(0.9 * len(ds))
    va = len(ds) - tr

    train, val = random_split(
        ds,
        [tr, va],
        generator=torch.Generator().manual_seed(42)
    )

    loader = DataLoader(
        train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    vloader = DataLoader(
        val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    # ==================================================
    # MODELS
    # ==================================================

    G = Generator(z_dim=Z_DIM, w_dim=Z_DIM).to(DEVICE)
    D = PerceiverDiscriminator().to(DEVICE)

    G_ema = copy.deepcopy(G).eval()

    for p in G_ema.parameters():
        p.requires_grad = False

    optG = optim.Adam(G.parameters(), lr=LR_G, betas=BETAS)
    optD = optim.Adam(D.parameters(), lr=LR_D, betas=BETAS)

    fid = FrechetInceptionDistance(feature=2048).to(DEVICE)
    is_metric = InceptionScore().to(DEVICE)

    ppl_avg = torch.zeros(1, device=DEVICE)

    best_fid = 1e9
    step = 0

    # ==================================================
    # HISTORY
    # ==================================================

    d_hist = []
    g_hist = []

    metric_epochs = []
    fid_hist = []
    is_hist = []

    # ==================================================
    # TRAIN
    # ==================================================

    for epoch in range(EPOCHS):

        G.train()
        D.train()

        d_run = 0
        g_run = 0
        count = 0

        for i, (real, _) in enumerate(loader):

            real = real.to(DEVICE, non_blocking=True)
            bs = real.size(0)

            # ==========================================
            # TRAIN D (1 step now)
            # ==========================================

            optD.zero_grad(set_to_none=True)

            z = torch.randn(bs, Z_DIM, device=DEVICE)

            with torch.no_grad():
                fake = G(z)

            real.requires_grad_(True)

            with autocast("cuda", enabled=AMP):

                dr = D(real).view(-1)
                df = D(fake).view(-1)

                dloss = d_hinge_loss(dr, df)

            if step % 32 == 0:

                grad = autograd.grad(
                    dr.sum(),
                    real,
                    create_graph=True
                )[0]

                r1 = grad.pow(2).reshape(bs, -1).sum(1).mean()

                dloss = dloss + (R1_GAMMA / 2) * r1

            scaler.scale(dloss).backward()

            scaler.unscale_(optD)
            torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)

            scaler.step(optD)
            scaler.update()
            if i % 20 == 0:
                print(
                    f"Real:{dr.mean().item():.3f} "
                    f"Fake:{df.mean().item():.3f}"
                    )

            # ==========================================
            # TRAIN G
            # ==========================================

            optG.zero_grad(set_to_none=True)

            z = torch.randn(bs, Z_DIM, device=DEVICE)

            with autocast("cuda", enabled=AMP):
                fake = G(z)
                pred = D(fake).view(-1)
                gloss = g_hinge_loss(pred)

            if step % 64 == 0:

                pl_bs = max(4, bs // 4)

                z2 = torch.randn(pl_bs, Z_DIM, device=DEVICE)

                w = G.mapping(z2)
                w = w.unsqueeze(1).repeat(1, 10, 1)

                img = G.synthesis(w)

                noise = torch.randn_like(img) / math.sqrt(64 * 64)

                grads = autograd.grad(
                    (img * noise).sum(),
                    w,
                    create_graph=True
                )[0]

                lens = torch.sqrt(
                    grads.pow(2).sum(dim=(1, 2)) + 1e-8
                )

                ppl_avg = (
                    0.99 * ppl_avg +
                    0.01 * lens.mean().detach()
                )

                gloss = gloss + (
                    ((lens - ppl_avg) ** 2).mean()
                    * PL_WEIGHT
                )

            scaler.scale(gloss).backward()
            scaler.step(optG)
            scaler.update()

            # ==========================================
            # EMA
            # ==========================================

            with torch.no_grad():
                for pe, p in zip(G_ema.parameters(), G.parameters()):
                    pe.data.mul_(0.999).add_(p.data, alpha=0.001)

            step += 1

            d_run += dloss.item()
            g_run += gloss.item()
            count += 1

            if i % 20 == 0:
                print(
                    f"E{epoch+1}/{EPOCHS} "
                    f"B{i}/{len(loader)} "
                    f"D:{dloss.item():.3f} "
                    f"G:{gloss.item():.3f}"
                )

        d_hist.append(d_run / count)
        g_hist.append(g_run / count)

        # ==========================================
        # SAVE SAMPLES
        # ==========================================

        with torch.no_grad():
            z = torch.randn(64, Z_DIM, device=DEVICE)
            fake = G_ema(z)
            utils.save_image(
                fake,
                f"{OUT}/samples/epoch_{epoch+1}.png",
                nrow=8,
                normalize=True,
                value_range=(-1,1)
                )

        # ==========================================
        # METRICS EVERY 3 EPOCHS
        # ==========================================

        if (epoch + 1) % 3 == 0:

            fid.reset()
            is_metric.reset()

            with torch.no_grad():

                for j, (real, _) in enumerate(vloader):

                    if j == 30:   # more reliable than 8
                        break

                    real = real.to(DEVICE)

                    fake = G_ema(
                        torch.randn(real.size(0), Z_DIM, device=DEVICE)
                    )

                    real_img = ((real + 1) / 2 * 255).byte()
                    fake_img = ((fake + 1) / 2 * 255).byte()

                    fid.update(real_img, real=True)
                    fid.update(fake_img, real=False)

                    is_metric.update(fake_img)

            fid_score = float(fid.compute())
            is_mean, is_std = is_metric.compute()

            print(f"FID : {fid_score:.4f}")
            print(f"IS  : {float(is_mean):.4f} ± {float(is_std):.4f}")

            metric_epochs.append(epoch + 1)
            fid_hist.append(fid_score)
            is_hist.append(float(is_mean))

            if fid_score < best_fid:
                best_fid = fid_score

                torch.save(
                    {
                        "G_ema": G_ema.state_dict(),
                        "fid": fid_score,
                        "epoch": epoch + 1
                    },
                    f"{OUT}/ckpt/best.pth"
                )

        save_graphs(
            OUT,
            d_hist,
            g_hist,
            metric_epochs,
            fid_hist,
            is_hist
        )

    print("Done")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()