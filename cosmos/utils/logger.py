"""
Logging utility — wraps console output and optional W&B.

Usage:
    log = Logger(project="cosmos", run_name="tokenizer_v1", use_wandb=True)
    log.info("training started")
    log.log({"train/loss": 0.4, "train/l1": 0.4}, step=1)
    log.log_images("recon", x, x_hat, step=1)   # x: [B,3,T,H,W]
    log.finish()
"""

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class Logger:
    def __init__(
        self,
        run_name:  str,
        config:    object = None,   # any dataclass — logged as wandb config
        use_wandb: bool   = False,
        log_dir:   Path   = Path("local_storage/logs"),
    ):
        self.use_wandb = use_wandb
        self._t0       = time.time()

        log_dir.mkdir(parents=True, exist_ok=True)

        if use_wandb:
            import wandb
            cfg_dict = asdict(config) if config is not None else {}
            cfg_dict = {k: str(v) if isinstance(v, Path) else v for k, v in cfg_dict.items()}
            self._wandb = wandb
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "cosmos"),
                entity=os.environ.get("WANDB_ENTITY"),
                name=run_name,
                config=cfg_dict,
                dir=os.environ.get("WANDB_DIR"),
            )
        else:
            self._wandb = None

        self.info(f"run: {run_name}  wandb={'on' if use_wandb else 'off'}")

    # ── console ───────────────────────────────────────────────────────────────

    def info(self, msg: str) -> None:
        elapsed = time.time() - self._t0
        print(f"[{elapsed:7.1f}s] {msg}", flush=True)

    # ── scalar metrics ────────────────────────────────────────────────────────

    def log(self, metrics: dict, step: int) -> None:
        if self._wandb:
            self._wandb.log(metrics, step=step)

    def log_train(self, log_dict: dict, step: int, epoch: int, lr: float) -> None:
        scalars = {f"train/{k}": v for k, v in log_dict.items()}
        scalars["train/lr"]    = lr
        scalars["train/epoch"] = epoch
        self.log(scalars, step=step)
        self.info(
            f"epoch {epoch:>3d}  step {step:>6d}  "
            + "  ".join(f"{k}={v:.4f}" for k, v in log_dict.items())
        )

    def log_val(self, val_loss: float, step: int, epoch: int) -> None:
        self.log({"val/loss": val_loss}, step=step)
        self.info(f"epoch {epoch:>3d} ── val_loss={val_loss:.4f}")

    # ── image logging ─────────────────────────────────────────────────────────

    def log_images(
        self,
        tag:   str,
        x:     torch.Tensor,   # [B, 3, T, H, W] in [-1, 1]
        x_hat: torch.Tensor,
        step:  int,
        max_clips: int = 4,
    ) -> None:
        if not self._wandb:
            return
        B = min(x.size(0), max_clips)
        T = x.size(2)
        rows = []
        for b in range(B):
            for clip, label in [(x, "gt"), (x_hat, "recon")]:
                frames = []
                for t in range(T):
                    frame = clip[b, :, t]                       # [3, H, W]
                    frame = (frame * 0.5 + 0.5).clamp(0, 1)    # [0, 1]
                    frames.append(frame.cpu().numpy())
                # concat frames horizontally → [3, H, T*W]
                row = np.concatenate(frames, axis=2)
                rows.append(self._wandb.Image(
                    row.transpose(1, 2, 0),   # H x T*W x 3
                    caption=f"clip{b} {label}",
                ))
        self._wandb.log({tag: rows}, step=step)

    # ── system info ───────────────────────────────────────────────────────────

    def log_system(self, model: torch.nn.Module) -> None:
        n_params = sum(p.numel() for p in model.parameters())
        self.info(f"model params: {n_params:,}")
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            self.info(f"GPU: {torch.cuda.get_device_name(dev)}"
                      f"  VRAM: {torch.cuda.get_device_properties(dev).total_memory // 1024**2} MB")

    # ── finish ────────────────────────────────────────────────────────────────

    def finish(self) -> None:
        elapsed = time.time() - self._t0
        self.info(f"done in {elapsed:.1f}s ({elapsed/60:.1f}min)")
        if self._wandb:
            self._wandb.finish()
