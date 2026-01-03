#!/usr/bin/env python3
"""
moment_run.py (Unified: Baseline + Continual Pretraining)

Supports:
- baseline: forecasting fine-tune only (no continual pretraining)
- sequential: continual pretraining (reconstruction) WITHOUT soft-masking, then forecasting fine-tune
- soft_masking: continual pretraining WITH soft-masking, then forecasting fine-tune
- all: run all three

Key features requested:
- --seq_len (context_len) and --pred_len configurable
- if seq_len < model expected length (typically 512), we pad + input_mask so head shapes match
- exposes continual-pretraining options implemented in the provided CL code
- avoids collisions with existing repo modules by using the `moment_cl` package namespace

Outputs:
  {result_dir}/{run_name}/{experiment}/
    config.json
    metrics.json
    artifacts/preds.npy, trues.npy
    models/*.pt
"""

from __future__ import annotations

import os
import warnings
import logging

# Silence most warnings/log spam (override by setting PYTHONWARNINGS=default)
os.environ.setdefault('PYTHONWARNINGS', 'ignore')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
warnings.filterwarnings('ignore')

try:
    from transformers import logging as hf_logging  # type: ignore
    hf_logging.set_verbosity_error()
except Exception:
    pass
logging.getLogger('transformers').setLevel(logging.ERROR)


import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from momentfm import MOMENTPipeline
from momentfm.utils.utils import control_randomness

from moment_cl import (
    DEFAULT_CONFIG,
    load_config,
    load_manufacturing_data,
    load_samyang_data,
    create_moment_dataloader,
    continual_pretrain,
    train_forecasting,
    evaluate_forecasting,
)
from moment_cl.utils import safe_save_model, print_memory_stats, clear_memory


def _to_jsonable(x):
    """Convert common non-JSON types (numpy/torch) into plain Python types."""
    try:
        import numpy as np
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
    except Exception:
        pass
    return str(x)

def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_to_jsonable)


def _merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("MOMENT unified runner (baseline + continual pretraining)")

    # Required target data
    p.add_argument("--data_path", required=True, type=str)
    p.add_argument("--target", required=True, type=str)
    p.add_argument("--minute_interval", type=int, default=15)

    # Forecasting lengths
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--pred_len", type=int, default=6)

    # Model id
    p.add_argument("--moment_model_id", type=str, default=None)

    # Output
    p.add_argument("--result_dir", type=str, default="result")
    p.add_argument("--run_name", type=str, default="moment_run")

    # Experiment selection
    p.add_argument("--experiment", type=str, default="baseline",
                   choices=["baseline", "sequential", "soft_masking", "all"])

    # Continual pretraining datasets
    p.add_argument("--pretrain_files", type=str, default=None,
                   help="Comma-separated manufacturing dataset CSVs (absolute or under data_dir)")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Directory for pretrain_files if they are not absolute paths")

    # Config file (optional)
    p.add_argument("--config_path", type=str, default=None)

    # Training (forecasting)
    p.add_argument("--fine_tune", action="store_true")
    p.add_argument("--moment_epochs", type=int, default=None)
    p.add_argument("--finetune_batch_size", type=int, default=None)
    p.add_argument("--finetune_lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--head_dropout", type=float, default=None)

    # Freezing
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--no_freeze_encoder", action="store_true")
    p.add_argument("--freeze_embedder", action="store_true")
    p.add_argument("--no_freeze_embedder", action="store_true")
    p.add_argument("--freeze_head", action="store_true")
    p.add_argument("--no_freeze_head", action="store_true")

    # Continual pretraining hyperparams
    p.add_argument("--pretrain_epochs", type=int, default=None)
    p.add_argument("--pretrain_batch_size", type=int, default=None)
    p.add_argument("--pretrain_lr", type=float, default=None)
    p.add_argument("--mask_ratio", type=float, default=None)
    p.add_argument("--pretrain_grad_clip", type=float, default=None)

    # Soft-masking options
    p.add_argument("--layer_to_mask", type=str, default=None, help="e.g., head,mlp")
    p.add_argument("--importance_samples", type=int, default=None)
    p.add_argument("--compute_cl_metrics", action="store_true")
    p.add_argument("--no_compute_cl_metrics", action="store_true")

    # Misc
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)

    return p.parse_args()


def _device(args: argparse.Namespace) -> torch.device:
    if args.device:
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _forecasting_model(cfg: Dict[str, Any], device: torch.device) -> MOMENTPipeline:
    model = MOMENTPipeline.from_pretrained(
        cfg["model_name"],
        model_kwargs={
            "task_name": "forecasting",
            "forecast_horizon": int(cfg["forecast_horizon"]),
            "head_dropout": float(cfg.get("head_dropout", 0.1)),
            "weight_decay": float(cfg.get("weight_decay", 0.01)),
            "freeze_encoder": bool(cfg.get("freeze_encoder", True)),
            "freeze_embedder": bool(cfg.get("freeze_embedder", True)),
            "freeze_head": bool(cfg.get("freeze_head", False)),
        },
    )
    model.init()
    return model.to(device)


def _reconstruction_model(cfg: Dict[str, Any], device: torch.device) -> MOMENTPipeline:
    model = MOMENTPipeline.from_pretrained(
        cfg["model_name"],
        model_kwargs={
            "task_name": "reconstruction",
            "freeze_encoder": False,
            "freeze_embedder": False,
        },
    )
    model.init()
    return model.to(device)


def _make_loaders(df, cfg):
    train_loader, train_dataset, target_idx = create_moment_dataloader(df, "train", cfg, shuffle=True, drop_last=True)
    val_loader, _, _ = create_moment_dataloader(df, "val", cfg, shuffle=False, drop_last=False)
    test_loader, _, _ = create_moment_dataloader(df, "test", cfg, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader, train_dataset, target_idx


def _run_baseline(cfg: Dict[str, Any], device: torch.device, data_dir: str, data_path: str, target: str, out_root: Path):
    out_dir = out_root / "baseline"
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    df, target_idx = load_samyang_data(data_dir, data_path, target)
    cfg["target_column"] = target

    train_loader, val_loader, test_loader, train_dataset, target_idx = _make_loaders(df, cfg)

    model = _forecasting_model(cfg, device)
    if cfg.get("do_finetune", True) and int(cfg.get("finetune_epochs", 0)) > 0:
        model = train_forecasting(model, train_loader, val_loader, cfg, device, target_idx, out_dir, model_name="baseline_forecasting")

    metrics, preds, trues = evaluate_forecasting(model, test_loader, device, target_idx, y_scaler=getattr(train_dataset, "y_scaler", None))

    np.save(out_dir / "artifacts" / "preds.npy", preds)
    np.save(out_dir / "artifacts" / "trues.npy", trues)
    _write_json(out_dir / "metrics.json", metrics)
    _write_json(out_dir / "config.json", cfg)
    safe_save_model(model, out_dir / "models" / "forecasting.pt", "baseline forecasting model")
    return {"metrics": metrics, "out_dir": str(out_dir)}


def _transfer_pretrained(pre_model: MOMENTPipeline, fore_model: MOMENTPipeline) -> MOMENTPipeline:
    pre_state = pre_model.state_dict()
    fore_state = fore_model.state_dict()
    for k in list(fore_state.keys()):
        if ("encoder" in k) or ("embed" in k):
            if k in pre_state:
                fore_state[k] = pre_state[k]
    fore_model.load_state_dict(fore_state)
    return fore_model


def _run_cl(cfg: Dict[str, Any], device: torch.device, arrays: List[np.ndarray], data_dir: str, data_path: str, target: str, out_root: Path, exp: str, use_soft_masking: bool):
    out_dir = out_root / exp
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    cfg["use_soft_masking"] = bool(use_soft_masking)

    pre_model = _reconstruction_model(cfg, device)
    pre_model = continual_pretrain(pre_model, arrays, cfg, device, out_dir)
    safe_save_model(pre_model, out_dir / "models" / "pretrained.pt", f"{exp} pretrained model")

    fore_model = _forecasting_model(cfg, device)
    fore_model = _transfer_pretrained(pre_model, fore_model).to(device)

    df, target_idx = load_samyang_data(data_dir, data_path, target)
    cfg["target_column"] = target
    train_loader, val_loader, test_loader, train_dataset, target_idx = _make_loaders(df, cfg)

    if cfg.get("do_finetune", True) and int(cfg.get("finetune_epochs", 0)) > 0:
        fore_model = train_forecasting(fore_model, train_loader, val_loader, cfg, device, target_idx, out_dir, model_name=f"{exp}_forecasting")

    metrics, preds, trues = evaluate_forecasting(fore_model, test_loader, device, target_idx, y_scaler=getattr(train_dataset, "y_scaler", None))

    np.save(out_dir / "artifacts" / "preds.npy", preds)
    np.save(out_dir / "artifacts" / "trues.npy", trues)
    _write_json(out_dir / "metrics.json", metrics)
    _write_json(out_dir / "config.json", cfg)
    safe_save_model(fore_model, out_dir / "models" / "forecasting.pt", f"{exp} forecasting model")
    return {"metrics": metrics, "out_dir": str(out_dir)}


def main():
    args = _parse_args()

    cfg = dict(DEFAULT_CONFIG)
    if args.config_path:
        cfg = _merge(cfg, load_config(args.config_path))

    data_path = args.data_path
    data_dir = args.data_dir or str(Path(data_path).parent)

    overrides = {
        "seed": args.seed,
        "data_dir": data_dir,
        "target_column": args.target,
        "minute_interval": int(args.minute_interval),

        "model_name": args.moment_model_id,

        "context_length": int(args.seq_len),
        "forecast_horizon": int(args.pred_len),

        "finetune_epochs": int(args.moment_epochs) if args.moment_epochs is not None else None,
        "finetune_batch_size": args.finetune_batch_size,
        "finetune_lr": args.finetune_lr,
        "weight_decay": args.weight_decay,
        "head_dropout": args.head_dropout,

        "pretrain_epochs": args.pretrain_epochs,
        "pretrain_batch_size": args.pretrain_batch_size,
        "pretrain_lr": args.pretrain_lr,
        "mask_ratio": args.mask_ratio,
        "pretrain_grad_clip": args.pretrain_grad_clip,

        "importance_samples": args.importance_samples,
    }

    if args.layer_to_mask:
        overrides["layer_to_mask"] = [s.strip() for s in args.layer_to_mask.split(",") if s.strip()]

    if args.compute_cl_metrics:
        overrides["compute_cl_metrics"] = True
    if args.no_compute_cl_metrics:
        overrides["compute_cl_metrics"] = False

    if args.freeze_encoder:
        overrides["freeze_encoder"] = True
    if args.no_freeze_encoder:
        overrides["freeze_encoder"] = False
    if args.freeze_embedder:
        overrides["freeze_embedder"] = True
    if args.no_freeze_embedder:
        overrides["freeze_embedder"] = False
    if args.freeze_head:
        overrides["freeze_head"] = True
    if args.no_freeze_head:
        overrides["freeze_head"] = False

    cfg = _merge(cfg, overrides)

    if not cfg.get("model_name"):
        cfg["model_name"] = DEFAULT_CONFIG["model_name"]

    # finetune epochs defaulting: if --fine_tune set and not specified, use default config value
    if cfg.get("finetune_epochs") is None:
        cfg["finetune_epochs"] = int(DEFAULT_CONFIG.get("finetune_epochs", 1)) if args.fine_tune else 0
    cfg["do_finetune"] = bool(args.fine_tune)

    # seed
    control_randomness(int(cfg.get("seed", 14)))

    device = _device(args)
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print_memory_stats("System ")

    out_root = Path(args.result_dir) / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)
    _write_json(out_root / "config_merged.json", cfg)

    exp = args.experiment
    want_baseline = exp in ("baseline", "all")
    want_seq = exp in ("sequential", "all")
    want_soft = exp in ("soft_masking", "all")

    arrays: List[np.ndarray] = []
    if want_seq or want_soft:
        pretrain_files = []
        if args.pretrain_files:
            pretrain_files = [s.strip() for s in args.pretrain_files.split(",") if s.strip()]
        else:
            pretrain_files = list(cfg.get("pretrain_files", []))
        if not pretrain_files:
            raise SystemExit("sequential/soft_masking requires --pretrain_files (comma-separated) or config.pretrain_files")
        arrays = load_manufacturing_data(data_dir, pretrain_files)

    results: Dict[str, Any] = {}
    try:
        if want_soft:
            cfg_soft = dict(cfg)
            results["soft_masking"] = _run_cl(cfg_soft, device, arrays, data_dir, data_path, args.target, out_root, "soft_masking", True)
            clear_memory()
        if want_seq:
            cfg_seq = dict(cfg)
            cfg_seq["use_soft_masking"] = False
            results["sequential"] = _run_cl(cfg_seq, device, arrays, data_dir, data_path, args.target, out_root, "sequential", False)
            clear_memory()
        if want_baseline:
            cfg_base = dict(cfg)
            results["baseline"] = _run_baseline(cfg_base, device, data_dir, data_path, args.target, out_root)
            clear_memory()
    finally:
        _write_json(out_root / "results_summary.json", results)

    print("Done. Results saved under:", out_root)


if __name__ == "__main__":
    main()
