"""
Overfit one batch to verify the tokenizer can learn.

A healthy run shows loss dropping toward <0.03 within ~50k steps and
reconstructions that visually match the input.

Resume from a previous checkpoint:
    Set resume_ckpt in the YAML (or Config) to the .pt path, then run normally.
    The run continues from where it left off — same run directory, same step
    counter, same LR schedule position.

Run:
    python scripts/overfit_one_batch.py
    python scripts/overfit_one_batch.py --config configs/overfit_continuous.yaml
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from dataclasses import dataclass
from cosmos.datasets.video_dataset import VideoDataset
from cosmos.tokenizers.continuous_tokenizer import ContinuousTokenizer
from cosmos.tokenizers.losses import TokenizerLoss
from cosmos.utils.viz import save_recon_grid, save_recon_gif
from cosmos.utils.logger import Logger
from cosmos.utils.run import make_run_dir, make_run_name, load_config


@dataclass
class Config:
    # data
    manifest:             Path      = Path("local_storage/dataset/bridgedata_clips/train.txt")
    device:               str       = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size:           int       = 4
    # model
    latent_channels:      int       = 16
    base_channels:        int       = 64
    temporal_compression: int       = 4
    use_attention:        bool      = False
    conv_type:            str       = "factorized"
    # loss
    use_perceptual:       bool      = True
    use_flow:             bool      = False
    use_gram:             bool      = False
    lambda_l1:            float     = 1.0
    lambda_perc:          float     = 0.1
    lambda_flow:          float     = 0.05
    lambda_gram:          float     = 0.02
    # train
    lr:                   float     = 1e-4
    lr_min_factor:        float     = 0.01
    steps:                int       = 50000
    grad_clip:            float     = 1.0
    resume_ckpt:          Path|None = None
    # log
    log_every:            int       = 1000
    viz_every:            int       = 10000
    use_wandb:            bool      = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None,
                   help="YAML config file (e.g. configs/overfit_continuous.yaml)")
    return p.parse_args()


def main():
    args = parse_args()
    CFG  = Config()
    if args.config:
        load_config(args.config, CFG)

    # ── run directory ─────────────────────────────────────────────────────────
    if CFG.resume_ckpt is not None:
        resume_ckpt = Path(CFG.resume_ckpt)
        run_dir  = resume_ckpt.parents[1]
        run_name = run_dir.name
        out_dir  = run_dir / "recon"
        ckpt_dir = run_dir / "ckpts"
        out_dir.mkdir(exist_ok=True)
        ckpt_dir.mkdir(exist_ok=True)
        log = Logger(run_name=run_name, config=CFG, use_wandb=CFG.use_wandb)
        log.info(f"resuming run: {run_dir}")
    else:
        run_name = make_run_name("overfit_continuous", CFG.manifest)
        dirs     = make_run_dir(run_name, config=CFG, yaml_path=args.config)
        out_dir  = dirs["recon_dir"]
        ckpt_dir = dirs["ckpt_dir"]
        log = Logger(run_name=run_name, config=CFG, use_wandb=CFG.use_wandb)
        log.info(f"run dir: {dirs['run_dir']}")

    # ── data ──────────────────────────────────────────────────────────────────
    ds      = VideoDataset(CFG.manifest)
    indices = list(range(min(CFG.batch_size, len(ds))))
    batch   = torch.stack([ds[i] for i in indices]).permute(0, 2, 1, 3, 4).to(CFG.device)
    log.info(f"batch shape: {tuple(batch.shape)}")

    # ── model + loss + optimiser ──────────────────────────────────────────────
    model = ContinuousTokenizer(
        latent_channels=CFG.latent_channels,
        base_channels=CFG.base_channels,
        temporal_compression=CFG.temporal_compression,
        use_attention=CFG.use_attention,
        conv_type=CFG.conv_type,
    ).to(CFG.device)

    criterion = TokenizerLoss(
        use_perceptual=CFG.use_perceptual,
        use_flow=CFG.use_flow,
        use_gram=CFG.use_gram,
        lambda_l1=CFG.lambda_l1,
        lambda_perc=CFG.lambda_perc,
        lambda_flow=CFG.lambda_flow,
        lambda_gram=CFG.lambda_gram,
    ).to(CFG.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.steps, eta_min=CFG.lr * CFG.lr_min_factor
    )

    # ── resume ────────────────────────────────────────────────────────────────
    start_step = 0
    if CFG.resume_ckpt is not None:
        ckpt = torch.load(CFG.resume_ckpt, map_location=CFG.device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        log.info(f"resumed from step {start_step}  lr={optimizer.param_groups[0]['lr']:.2e}")

    log.log_system(model)
    log.info(f"tc={CFG.temporal_compression}  base_ch={CFG.base_channels}  latent_ch={CFG.latent_channels}")

    # ── overfit loop ──────────────────────────────────────────────────────────
    model.train()
    for step in range(start_step + 1, start_step + CFG.steps + 1):
        optimizer.zero_grad()
        x_hat, z = model(batch)
        loss, loss_dict = criterion(x_hat, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % CFG.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            log.log_train(loss_dict, step=step, epoch=1, lr=lr)

        if step % CFG.viz_every == 0:
            model.eval()
            with torch.no_grad():
                x_hat_viz, _ = model(batch)
            save_recon_grid(batch.cpu(), x_hat_viz.cpu(), out_dir / f"step_{step:06d}.png")
            save_recon_gif(batch[0].cpu(), x_hat_viz[0].cpu(), out_dir / f"step_{step:06d}.gif")
            log.log_images("recon", batch, x_hat_viz, step=step)
            model.train()

    # ── final viz + checkpoint ────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        x_hat_final, z = model(batch)

    final_step = start_step + CFG.steps
    save_recon_grid(batch.cpu(), x_hat_final.cpu(), out_dir / "final.png")
    save_recon_gif(batch[0].cpu(), x_hat_final[0].cpu(), out_dir / "final.gif")
    log.log_images("recon", batch, x_hat_final, step=final_step)
    log.info(f"latent shape: {tuple(z.shape)}")

    ckpt_path = ckpt_dir / f"{run_name}_step{final_step:06d}.pt"
    torch.save({
        "step":      final_step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, ckpt_path)
    log.info(f"checkpoint → {ckpt_path}")
    log.finish()


if __name__ == "__main__":
    main()
