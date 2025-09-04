#!/usr/bin/env python3
"""
Evaluate a directory of CTC checkpoints on a validation set and summarize metrics.

Outputs:
- results/ctc_checkpoint_sweep.json
- results/ctc_checkpoint_sweep.csv

Metrics:
- epoch, val_loss (if present in checkpoint), accuracy, LER, total_lev, total_gt_len, checkpoint_path
"""

import os
import csv
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch

from Src.Data.DataLoader import CaptchaDataLoader
from Src.Model.CTC_Model import CaptchaSolverModel
from Src.Utils.Evaluations import (
    get_idx_to_char,
    get_char_to_cat,
    summarize_metrics,
)


def _infer_num_classes(ckpt: dict) -> int:
    """
    Infer output class count from checkpoint config or classifier weights.

    Args:
        ckpt (dict): Loaded checkpoint dictionary.

    Returns:
        int: Number of classes including the CTC blank.

    Raises:
        RuntimeError: If num_classes cannot be inferred.
    """
    cfg = ckpt.get("model_config") or {}
    if isinstance(cfg, dict) and "num_classes" in cfg:
        return int(cfg["num_classes"])
    sd = ckpt.get("model_state_dict", {})
    for k in ["head.fc.weight", "fc.weight", "classifier.weight", "head.pred_conv.weight", "head.pred.weight"]:
        if k in sd:
            return int(sd[k].shape[0])
    raise RuntimeError("num_classes not found; store it in checkpoint['model_config']['num_classes'].")


def _load_model(ckpt_path: str, device: torch.device) -> dict:
    """
    Load a checkpoint, build the model, and prepare mappings.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        device (torch.device): Target device.

    Returns:
        dict: Dictionary with model, mappings, blank index, printable size, and metadata.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    num_classes = _infer_num_classes(ckpt)
    model = CaptchaSolverModel(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    blank_idx = num_classes - 1
    printable_size = num_classes - 1
    idx_to_char = get_idx_to_char(num_classes)
    meta = {
        "epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss"),
        "best_loss": ckpt.get("best_loss"),
        "num_classes": num_classes,
    }
    return {
        "model": model,
        "idx_to_char": idx_to_char,
        "blank_idx": blank_idx,
        "printable_size": printable_size,
        "meta": meta,
    }


def _decode_row(row, blank_idx: int, printable_size: int, idx_to_char: dict) -> str:
    """
    Decode a single CTC path row by collapsing repeats and removing blanks.

    Args:
        row: Sequence of class indices.
        blank_idx (int): Index for CTC blank.
        printable_size (int): Upper bound on printable indices.
        idx_to_char (dict): Index-to-character mapping.

    Returns:
        str: Decoded string.
    """
    out, prev = [], None
    for k in row:
        k = int(k)
        if k == blank_idx:
            prev = k
            continue
        if 0 <= k < printable_size and k != prev:
            out.append(idx_to_char[k])
        prev = k
    return "".join(out)


def _predict_batch(model, images: torch.Tensor, blank_idx: int, printable_size: int, idx_to_char: dict) -> List[str]:
    """
    Predict a batch of images and decode CTC outputs.

    Args:
        model: Torch model.
        images (Tensor): Batch tensor (B,C,H,W).
        blank_idx (int): CTC blank index.
        printable_size (int): Upper bound on printable indices.
        idx_to_char (dict): Index-to-character mapping.

    Returns:
        List[str]: Decoded predictions.
    """
    with torch.no_grad():
        logits = model(images)
        if logits.dim() != 3:
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
        if logits.shape[0] == images.size(0):
            paths = logits.argmax(-1).cpu().numpy()
            return [_decode_row(row, blank_idx, printable_size, idx_to_char) for row in paths]
        T, B, _ = logits.shape
        paths = logits.argmax(-1).cpu().numpy()
        return [_decode_row(paths[:, b], blank_idx, printable_size, idx_to_char) for b in range(B)]


def _gather_metrics_for_checkpoint(
    ckpt_path: str,
    data_dir: str,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate one checkpoint on the dataset and return metric summary.

    Args:
        ckpt_path (str): Path to checkpoint.
        data_dir (str): Validation dataset directory.
        device (torch.device): Target device.
        max_batches (int | None): Optional cap on batches.

    Returns:
        dict: Metrics and metadata for the checkpoint.
    """
    bundle = _load_model(ckpt_path, device)
    model = bundle["model"]
    idx_to_char = bundle["idx_to_char"]
    blank_idx = bundle["blank_idx"]
    printable_size = bundle["printable_size"]
    meta = bundle["meta"]

    loader = CaptchaDataLoader(data_dir, batch_size=1, shuffle=False)
    preds, gts = [], []
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        images = batch["Image"].to(device)
        gt_field = batch.get("CaptchaString")
        gt = gt_field[0] if isinstance(gt_field, list) else batch.get("captcha_string", "")
        pred = _predict_batch(model, images, blank_idx, printable_size, idx_to_char)[0]
        if isinstance(gt, str) and len(gt) > 0:
            preds.append(pred)
            gts.append(gt)

    summary = summarize_metrics(preds, gts) if gts else {
        "samples": 0,
        "accuracy": 0.0,
        "ler": 0.0,
        "total_lev": 0,
        "total_gt_len": 0,
        "lev_div_20000": 0.0,
    }

    return {
        "epoch": meta.get("epoch"),
        "val_loss": meta.get("val_loss"),
        "best_loss": meta.get("best_loss"),
        "num_classes": meta.get("num_classes"),
        "accuracy": summary["accuracy"],
        "ler": summary["ler"],
        "total_lev": summary["total_lev"],
        "total_gt_len": summary["total_gt_len"],
        "lev_div_20000": summary["lev_div_20000"],
        "checkpoint_path": str(ckpt_path),
    }


def evaluate_checkpoints(
    checkpoint_dir: str,
    data_dir: str,
    pattern: str = "epoch_*.pth",
    device_str: str = "cuda",
    max_batches: Optional[int] = None,
    out_json: str = "results/ctc_checkpoint_sweep.json",
    out_csv: str = "results/ctc_checkpoint_sweep.csv",
) -> Dict[str, Any]:
    """
    Sweep checkpoints in a directory, evaluate each, and write JSON/CSV summaries.

    Args:
        checkpoint_dir (str): Directory containing checkpoints.
        data_dir (str): Validation dataset directory.
        pattern (str): Glob pattern to match checkpoints.
        device_str (str): Preferred device ("cuda" or "cpu").
        max_batches (int | None): Optional cap on batches per checkpoint.
        out_json (str): Output JSON path.
        out_csv (str): Output CSV path.

    Returns:
        dict: Summary with per-checkpoint metrics and best selection by LER.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    ckpts = sorted(glob.glob(str(Path(checkpoint_dir) / pattern)))
    results: List[Dict[str, Any]] = []

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    for ckpt_path in ckpts:
        row = _gather_metrics_for_checkpoint(ckpt_path, data_dir, device, max_batches=max_batches)
        results.append(row)
        print(f"Evaluated: {Path(ckpt_path).name} | LER={row['ler']:.4f} | Acc={row['accuracy']*100:.2f}% | val_loss={row.get('val_loss')}")

    best_by_ler = min(results, key=lambda r: r["ler"]) if results else None
    best_epoch = best_by_ler.get("epoch") if best_by_ler else None

    with open(out_json, "w", encoding="utf-8") as fj:
        json.dump({"results": results, "best_by_ler": best_by_ler}, fj, ensure_ascii=False, indent=2)

    with open(out_csv, "w", newline="", encoding="utf-8") as fc:
        writer = csv.DictWriter(
            fc,
            fieldnames=[
                "epoch",
                "val_loss",
                "best_loss",
                "num_classes",
                "accuracy",
                "ler",
                "total_lev",
                "total_gt_len",
                "lev_div_20000",
                "checkpoint_path",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nJSON written to: {os.path.abspath(out_json)}")
    print(f"CSV  written to: {os.path.abspath(out_csv)}")
    if best_by_ler:
        print(f"Best by LER: epoch={best_epoch}, LER={best_by_ler['ler']:.4f}, Acc={best_by_ler['accuracy']*100:.2f}%")
    else:
        print("No checkpoints matched the pattern.")
    return {"results": results, "best_by_ler": best_by_ler}


if __name__ == "__main__":
    checkpoint_dir = "./checkpoints"
    data_dir = "/home/hpc/v123be/v123be34/UTN-CV25-Captcha-Dataset/part2/val"
    evaluate_checkpoints(
        checkpoint_dir=checkpoint_dir,
        data_dir=data_dir,
        pattern="epoch_*.pth",
        device_str="cuda",
        max_batches=None,
        out_json="results/ctc_checkpoint_sweep.json",
        out_csv="results/ctc_checkpoint_sweep.csv",
    )
