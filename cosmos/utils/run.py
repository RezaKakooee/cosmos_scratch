"""
Run directory management + YAML config loader.

Creates a timestamped run folder and snapshots the source code into it.

Layout:
    local_storage/runs/
      2026_04_22_1427__tokenizer_v1__bridgedata_clips/
        recon/
        ckpts/
        code/          ← snapshot of cosmos/ + running script
          cosmos/
          scripts/
          config.json
"""

import json
import shutil
import yaml
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


def load_config(yaml_path: str, cfg_instance) -> object:
    """
    Load a nested YAML file and override matching fields on a dataclass instance.

    The YAML may use top-level sections (model, loss, train, log, data) — each
    section is a flat dict whose keys must match Config field names.  Unknown
    keys are silently ignored so you can add comments/sections freely.

    Path fields (manifest, resume_ckpt, train_manifest, val_manifest) are
    converted to Path objects automatically.  'null' in YAML becomes None.
    """
    with open(yaml_path) as f:
        d = yaml.safe_load(f) or {}

    # Flatten all nested sections into one dict
    flat: dict = {}
    for v in d.values():
        if isinstance(v, dict):
            flat.update(v)

    _path_fields = {"manifest", "resume_ckpt", "train_manifest", "val_manifest"}

    for k, v in flat.items():
        if not hasattr(cfg_instance, k):
            continue
        if k in _path_fields:
            setattr(cfg_instance, k, Path(v) if v is not None else None)
        else:
            setattr(cfg_instance, k, v)

    return cfg_instance


def _dataset_name(manifest_path: Path) -> str:
    """Extract a short dataset label from a manifest path."""
    # e.g. local_storage/dataset/bridgedata_clips/train.txt → bridgedata_clips
    return manifest_path.parent.name


def make_run_name(model_tag: str, manifest_path: Path) -> str:
    """
    Build a run name: 2026_04_22_1427__tokenizer_v1__bridgedata_clips
    """
    ts      = datetime.now().strftime("%Y_%m_%d_%H%M")
    dataset = _dataset_name(Path(manifest_path))
    return f"{ts}__{model_tag}__{dataset}"


def make_run_dir(
    run_name:  str,
    config:    object      = None,
    yaml_path: str | None  = None,
) -> dict[str, Path]:
    """
    Create the run directory tree and snapshot source code + config.

    Layout:
        <run_dir>/
          recon/
          ckpts/
          code/
            cosmos/        ← source snapshot
            scripts/
            config.json    ← all resolved Config fields as JSON
            config.yaml    ← original YAML file (if --config was passed)

    Returns a dict with keys: run_dir, recon_dir, ckpt_dir, code_dir
    """
    project_root = Path(__file__).parents[2]   # cosmos_scratch/
    run_dir   = project_root / "local_storage" / "runs" / run_name
    recon_dir = run_dir / "recon"
    ckpt_dir  = run_dir / "ckpts"
    code_dir  = run_dir / "code"

    for d in (recon_dir, ckpt_dir, code_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── code snapshot ─────────────────────────────────────────────────────────
    for src_name in ("cosmos", "scripts", "data"):
        src = project_root / src_name
        dst = code_dir / src_name
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # ── config snapshots ──────────────────────────────────────────────────────
    if config is not None:
        cfg_dict = asdict(config)
        cfg_dict = {k: str(v) if isinstance(v, Path) else v for k, v in cfg_dict.items()}
        (code_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))

    if yaml_path is not None:
        shutil.copy(yaml_path, code_dir / "config.yaml")

    return {"run_dir": run_dir, "recon_dir": recon_dir, "ckpt_dir": ckpt_dir, "code_dir": code_dir}
