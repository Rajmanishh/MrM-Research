import os
import copy
import multiprocessing
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt

# Metrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance

# Your Models
from stylegan.generator.generator import Generator
from stylegan.discriminator.cnn_discriminator import StyleGAN1Discriminator


# --------------------------------------------------
# Stable Weight Init
# --------------------------------------------------
def init_weights_stable(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# --------------------------------------------------
# Save Images + Graphs
# --------------------------------------------------
def save_research_artifacts(
    G_ema,
    history,
    epoch,
    sample_dir,
    graph_dir,
    device,
    z_dim
):
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    # Save sample grid
    with torch.no_grad():
        G_ema.eval()

        z = torch.randn(64, z_dim, device=device)
        fake = G_ema(z, style_mixing_prob=0)

        fake = (fake + 1) / 2
        fake = fake.clamp(0, 1)

        utils.save_image(
            fake,
            os.path.join(sample_dir, f"epoch_{epoch+1}.png"),
            nrow=8
        )

    # Save graphs
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axs[0].plot(history["d_loss"], label="D Loss")
    axs[0].plot(history["g_loss"], label="G Loss")
    axs[0].set_title("Loss Curves")
    axs[0].legend()

    # FID + KID
    if len(history["metric_epochs"]) > 0:

        ax2 = axs[1].twinx()

        axs[1].plot(
            history["metric_epochs"],
            history["fid"],
            "g-o",
            label="FID"
        )

        ax2.plot(
            history["metric_epochs"],
            history["kid"],
            "m--s",
            label="KID"
        )

        axs[1].set_title("FID / KID")
        axs[1].legend(loc="upper left")
        ax2.legend(loc="upper right")

        axs[2].plot(
            history["metric_epochs"],
            history["is"],
            color="orange",
            marker="^",
            label="IS"
        )

        axs[2].set_title("Inception Score")
        axs[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, "research_summary.png"))
    plt.close()


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("DEVICE :", DEVICE)

    if DEVICE == "cuda":
        print("GPU    :", torch.cuda.get_device_name(0))

    print("=" * 60)

    # ------------------------------------------
    # Config
    # ------------------------------------------
    BATCH_SIZE = 80
    EPOCHS = 20

    LR_G = 1e-4
    LR_D = 2e-4

    Z_DIM = 256
    STYLE_MIX = 0.2
    R1_GAMMA = 2.0
    METRIC_INTERVAL = 2

    BASE_DIR = "run_ffhq_fixed"
    SAMPLE_DIR = os.path.join(BASE_DIR, "samples")
    GRAPH_DIR = os.path.join(BASE_DIR, "graphs")

    os.makedirs(SAMPLE_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)

    torch.manual_seed(1337)

    # ------------------------------------------
    # Transform
    # ------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    # ------------------------------------------
    # Dataset (90 / 10)
    # ------------------------------------------
    DATA_PATH = "./data/ffhq"

    full_dataset = datasets.ImageFolder(
        root=DATA_PATH,
        transform=transform
    )

    total = len(full_dataset)

    train_size = int(0.90 * total)
    val_size = total - train_size

    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    print(f"Train : {train_size}")
    print(f"Val   : {val_size}")

    # ------------------------------------------
    # Models
    # ------------------------------------------
    G = Generator(
        z_dim=Z_DIM,
        w_dim=Z_DIM
    ).to(DEVICE)

    D = StyleGAN1Discriminator().to(DEVICE)

    G.apply(init_weights_stable)
    D.apply(init_weights_stable)

    G_ema = copy.deepcopy(G).eval()

    # ------------------------------------------
    # Optimizers
    # ------------------------------------------
    opt_G = optim.Adam(
        G.parameters(),
        lr=LR_G,
        betas=(0.0, 0.99)
    )

    opt_D = optim.Adam(
        D.parameters(),
        lr=LR_D,
        betas=(0.0, 0.99)
    )

    # ------------------------------------------
    # History
    # ------------------------------------------
    history = {
        "d_loss": [],
        "g_loss": [],
        "fid": [],
        "is": [],
        "kid": [],
        "metric_epochs": []
    }

    # ==================================================
    # Training Loop
    # ==================================================
    for epoch in range(EPOCHS):

        G.train()
        D.train()

        epoch_d = 0
        epoch_g = 0

        for i, (real_imgs, _) in enumerate(train_loader):

            real_imgs = real_imgs.to(DEVICE)
            bs = real_imgs.size(0)

            # ----------------------------------
            # Train D
            # ----------------------------------
            z = torch.randn(bs, Z_DIM, device=DEVICE)

            fake = G(
                z,
                style_mixing_prob=STYLE_MIX
            )

            d_real = D(real_imgs)
            d_fake = D(fake.detach())

            d_loss = (
                F.softplus(-d_real).mean()
                + F.softplus(d_fake).mean()
            )

            # R1 regularization
            if i % 16 == 0:

                real_imgs.requires_grad_(True)

                real_pred = D(real_imgs)

                grad_real = autograd.grad(
                    outputs=real_pred.sum(),
                    inputs=real_imgs,
                    create_graph=True
                )[0]

                grad_penalty = (
                    grad_real.pow(2)
                    .reshape(bs, -1)
                    .sum(1)
                    .mean()
                )

                d_loss = d_loss + (R1_GAMMA / 2) * grad_penalty

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # ----------------------------------
            # Train G (fresh batch)
            # ----------------------------------
            z = torch.randn(bs, Z_DIM, device=DEVICE)

            fake = G(
                z,
                style_mixing_prob=STYLE_MIX
            )

            g_loss = F.softplus(-D(fake)).mean()

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            # ----------------------------------
            # EMA
            # ----------------------------------
            with torch.no_grad():
                for p_ema, p in zip(
                    G_ema.parameters(),
                    G.parameters()
                ):
                    p_ema.data.mul_(0.995).add_(
                        p.data,
                        alpha=0.005
                    )

            epoch_d += d_loss.item()
            epoch_g += g_loss.item()

            if i % 10 == 0:
                print(
                    f"E[{epoch+1}/{EPOCHS}] "
                    f"B[{i}/{len(train_loader)}] "
                    f"D:{d_loss.item():.4f} "
                    f"G:{g_loss.item():.4f}",
                    end="\r"
                )

        # ------------------------------------------
        # Epoch stats
        # ------------------------------------------
        epoch_d /= len(train_loader)
        epoch_g /= len(train_loader)

        history["d_loss"].append(epoch_d)
        history["g_loss"].append(epoch_g)

        print()
        print(f"\nEpoch {epoch+1}")
        print(f"D Loss : {epoch_d:.4f}")
        print(f"G Loss : {epoch_g:.4f}")

        # ------------------------------------------
        # Metrics
        # ------------------------------------------
        if (epoch + 1) % METRIC_INTERVAL == 0:

            torch.cuda.empty_cache()

            fid_m = FrechetInceptionDistance(
                feature=64
            ).to(DEVICE)

            is_m = InceptionScore().to(DEVICE)

            kid_m = KernelInceptionDistance(
                subset_size=50
            ).to(DEVICE)

            G_ema.eval()

            with torch.no_grad():

                for val_batch, _ in val_loader:

                    val_batch = val_batch.to(DEVICE)

                    # Real
                    real_uint8 = (
                        ((val_batch + 1) / 2) * 255
                    ).clamp(0, 255).to(torch.uint8)

                    # Fake
                    z = torch.randn(
                        val_batch.size(0),
                        Z_DIM,
                        device=DEVICE
                    )

                    fake = G_ema(
                        z,
                        style_mixing_prob=0
                    )

                    fake_uint8 = (
                        ((fake + 1) / 2) * 255
                    ).clamp(0, 255).to(torch.uint8)

                    fid_m.update(real_uint8, real=True)
                    fid_m.update(fake_uint8, real=False)

                    is_m.update(fake_uint8)

                    kid_m.update(real_uint8, real=True)
                    kid_m.update(fake_uint8, real=False)

            fid_score = fid_m.compute().item()
            is_score = is_m.compute()[0].item()

            kid_score, _ = kid_m.compute()
            kid_score = kid_score.item()

            history["fid"].append(fid_score)
            history["is"].append(is_score)
            history["kid"].append(kid_score)
            history["metric_epochs"].append(epoch + 1)

            print(
                f"FID: {fid_score:.2f} | "
                f"IS: {is_score:.2f} | "
                f"KID: {kid_score:.4f}"
            )

        # ------------------------------------------
        # Save Outputs
        # ------------------------------------------
        save_research_artifacts(
            G_ema,
            history,
            epoch,
            SAMPLE_DIR,
            GRAPH_DIR,
            DEVICE,
            Z_DIM
        )

    print("\nTraining Complete.")


# --------------------------------------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()