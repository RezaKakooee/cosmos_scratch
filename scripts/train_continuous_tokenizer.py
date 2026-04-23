"""
Full training loop for ContinuousTokenizer.

Loss schedule — enable one at a time once the previous stage converges:
  Stage 1: L1 only               (use_perceptual=False, use_flow=False, use_gram=False)
  Stage 2: L1 + perceptual       (use_perceptual=True)
  Stage 3: L1 + perceptual + flow
  Stage 4: full loss stack

Run:
    python scripts/train_continuous_tokenizer.py
    python scripts/train_continuous_tokenizer.py --config configs/train_continuous.yaml
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from dataclasses import dataclass
from cosmos.datasets.video_dataset import make_dataloaders
from cosmos.tokenizers.continuous_tokenizer import ContinuousTokenizer
from cosmos.tokenizers.losses import TokenizerLoss
from cosmos.utils.viz import save_recon_grid, save_recon_gif
from cosmos.utils.logger import Logger
from cosmos.utils.run import make_run_dir, make_run_name, load_config


@dataclass
class Config:
    # data
    train_manifest:       Path  = Path("local_storage/dataset/bridgedata_clips/train.txt")
    val_manifest:         Path  = Path("local_storage/dataset/bridgedata_clips/val.txt")
    device:               str   = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size:           int   = 8
    num_workers:          int   = 4
    # model
    latent_channels:      int   = 16
    base_channels:        int   = 64
    temporal_compression: int   = 4
    use_attention:        bool  = True
    conv_type:            str   = "factorized"
    # loss
    use_perceptual:       bool  = False
    use_flow:             bool  = False
    use_gram:             bool  = False
    lambda_l1:            float = 1.0
    lambda_perc:          float = 0.1
    lambda_flow:          float = 0.05
    lambda_gram:          float = 0.02
    # train
    lr:                   float = 1e-4
    epochs:               int   = 20
    grad_clip:            float = 1.0
    log_every:            int   = 50
    viz_every:            int   = 200
    ckpt_every:           int   = 500
    # log
    use_wandb:            bool  = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None,
                   help="YAML config file (e.g. configs/train_continuous.yaml)")
    return p.parse_args()


def save_checkpoint(model, optimizer, epoch, step, path):
    torch.save({
        "epoch":     epoch,
        "step":      step,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)


def run_val(model, val_dl, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in val_dl:
            x = batch.permute(0, 2, 1, 3, 4).to(device)
            x_hat, _ = model(x)
            _, loss_dict = criterion(x_hat, x)
            total += loss_dict["total"]
    model.train()
    return total / len(val_dl)


def main():
    args = parse_args()
    CFG  = Config()
    if args.config:
        load_config(args.config, CFG)

    run_name = make_run_name("continuous_tokenizer", CFG.train_manifest)
    dirs     = make_run_dir(run_name, config=CFG, yaml_path=args.config)
    out_dir  = dirs["recon_dir"]
    ckpt_dir = dirs["ckpt_dir"]

    log = Logger(run_name=run_name, config=CFG, use_wandb=CFG.use_wandb)
    log.info(f"run dir: {dirs['run_dir']}")

    train_dl, val_dl = make_dataloaders(
        CFG.train_manifest, CFG.val_manifest,
        batch_size=CFG.batch_size, num_workers=CFG.num_workers,
    )
    log.info(f"train clips: {len(train_dl.dataset)}  val clips: {len(val_dl.dataset)}")

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

    log.log_system(model)
    log.info(f"tc={CFG.temporal_compression}  base_ch={CFG.base_channels}  latent_ch={CFG.latent_channels}")

    global_step = 0

    for epoch in range(1, CFG.epochs + 1):
        for batch in train_dl:
            x = batch.permute(0, 2, 1, 3, 4).to(CFG.device)

            optimizer.zero_grad()
            x_hat, z = model(x)
            loss, loss_dict = criterion(x_hat, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            optimizer.step()
            global_step += 1

            if global_step % CFG.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                log.log_train(loss_dict, step=global_step, epoch=epoch, lr=lr)

            if global_step % CFG.viz_every == 0:
                model.eval()
                with torch.no_grad():
                    x_hat_viz, _ = model(x)
                save_recon_grid(x.cpu(), x_hat_viz.cpu(),
                                out_dir / f"step_{global_step:06d}.png")
                save_recon_gif(x[0].cpu(), x_hat_viz[0].cpu(),
                               out_dir / f"step_{global_step:06d}.gif")
                log.log_images("recon", x, x_hat_viz, step=global_step)
                model.train()

            if global_step % CFG.ckpt_every == 0:
                ckpt_path = ckpt_dir / f"continuous_tokenizer_step{global_step:06d}.pt"
                save_checkpoint(model, optimizer, epoch, global_step, ckpt_path)
                log.info(f"checkpoint → {ckpt_path}")

        val_loss = run_val(model, val_dl, criterion, CFG.device)
        log.log_val(val_loss, step=global_step, epoch=epoch)

    ckpt_path = ckpt_dir / "continuous_tokenizer_final.pt"
    save_checkpoint(model, optimizer, CFG.epochs, global_step, ckpt_path)
    log.info(f"final checkpoint → {ckpt_path}")
    log.finish()


if __name__ == "__main__":
    main()
