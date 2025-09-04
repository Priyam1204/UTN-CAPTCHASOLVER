#!/usr/bin/env python3
"""
String-only evaluator for CTC-style CAPTCHA models.
Auto-detects num_classes, performs greedy CTC decoding, prints metrics, and writes a single dataset-style JSON.
"""

import os
import json
from pathlib import Path

import torch

from Src.Data.DataLoader import CaptchaDataLoader
from Src.Model.CTC_Model import CaptchaSolverModel
from Src.Utils.Evaluations import (
    get_idx_to_char,
    get_char_to_cat,
    build_json_entry,
    summarize_metrics,
    print_eval_summary,
)


class StringEvaluator:
    """
    Evaluate a CTC CAPTCHA model on a dataset directory and export predictions as a dataset-style JSON.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Load checkpoint, infer num_classes, and prepare model and character mappings.

        Args:
            model_path (str): Path to model checkpoint.
            device (str): Preferred device ("cuda" or "cpu").
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location=self.device)

        num_classes = None
        cfg = ckpt.get("model_config") or {}
        if isinstance(cfg, dict) and "num_classes" in cfg:
            num_classes = int(cfg["num_classes"])
        if num_classes is None:
            sd = ckpt.get("model_state_dict", {})
            for k in ["head.fc.weight", "fc.weight", "classifier.weight", "head.pred_conv.weight", "head.pred.weight"]:
                if k in sd:
                    num_classes = sd[k].shape[0]
                    break
            if num_classes is None:
                raise RuntimeError("Could not infer num_classes; store it in checkpoint['model_config']['num_classes'].")

        self.model = CaptchaSolverModel(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model.eval()

        self.blank_idx = num_classes - 1
        self.printable_size = num_classes - 1
        self.idx_to_char = get_idx_to_char(num_classes)
        self.char_to_cat = get_char_to_cat()

        print(f"Loaded model from {model_path}")
        print(f"Detected num_classes={num_classes} (blank_idx={self.blank_idx})")
        print(f"Epoch {ckpt.get('epoch','?')} | Best loss {ckpt.get('best_loss','?')}")

    def _decode_path_row(self, row, blank_idx: int) -> str:
        """
        Collapse repeats and drop blanks to produce a string.

        Args:
            row: Sequence of class indices for one sample.
            blank_idx (int): Index used for the CTC blank.

        Returns:
            str: Decoded string.
        """
        out, prev = [], None
        for k in row:
            k = int(k)
            if k == blank_idx:
                prev = k
                continue
            if 0 <= k < self.printable_size and k != prev:
                out.append(self.idx_to_char[k])
            prev = k
        return "".join(out)

    def predict_batch(self, images: torch.Tensor):
        """
        Run a forward pass and decode a batch of images.

        Args:
            images (Tensor): Batch tensor of shape (B,C,H,W).

        Returns:
            List[str]: Decoded predictions for the batch.
        """
        with torch.no_grad():
            logits = self.model(images)
            if logits.dim() != 3:
                raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

            if logits.shape[0] == images.size(0):
                paths = logits.argmax(-1).cpu().numpy()
                return [self._decode_path_row(row, self.blank_idx) for row in paths]

            T, B, _ = logits.shape
            paths = logits.argmax(-1).cpu().numpy()
            return [self._decode_path_row(paths[:, b], self.blank_idx) for b in range(B)]

    def _extract_image_id(self, batch, fallback_index: int) -> str:
        """
        Extract an image_id from batch metadata or use a numeric fallback.

        Args:
            batch (dict): DataLoader batch with path fields.
            fallback_index (int): Index to use if path is missing.

        Returns:
            str: Image identifier (filename stem or zero-padded index).
        """
        img_path = batch.get("ImagePath") or batch.get("image_path") or batch.get("Path") or ""
        if isinstance(img_path, list):
            img_path = img_path[0]
        return Path(img_path).stem if img_path else f"{fallback_index+1:06d}"

    def evaluate(
        self,
        data_dir: str,
        max_batches: int | None = None,
        json_path: str = "eval_results.json",
        json_string_source: str = "pred",
    ) -> str:
        """
        Evaluate the dataset, print summary metrics, and write a single JSON file of predictions.

        Args:
            data_dir (str): Dataset directory.
            max_batches (int | None): Optional cap on number of batches to process.
            json_path (str): Output JSON path.
            json_string_source (str): "pred" to export predictions, "gt" to export ground truth.

        Returns:
            str: Absolute path to the written JSON file.
        """
        loader = CaptchaDataLoader(data_dir, batch_size=1, shuffle=False)
        preds, gts, json_entries = [], [], []
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)

        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break

            image = batch["Image"].to(self.device)
            _, _, H, W = image.shape

            gt_field = batch.get("CaptchaString")
            gt = gt_field[0] if isinstance(gt_field, list) else batch.get("captcha_string", "")

            pred = self.predict_batch(image)[0]
            image_id = self._extract_image_id(batch, i)

            if isinstance(gt, str) and len(gt) > 0:
                preds.append(pred)
                gts.append(gt)

            captcha_string = pred if json_string_source.lower() == "pred" else (gt if isinstance(gt, str) else "")
            entry = build_json_entry(
                height=int(H),
                width=int(W),
                image_path=batch.get("ImagePath") if batch.get("ImagePath") else image_id,
                captcha_string=captcha_string,
                char_to_cat=self.char_to_cat,
                index=i,
            )
            json_entries.append(entry)

            print(f"[{i+1:05d}] ID={image_id}  Pred={pred}  GT={gt if isinstance(gt, str) else ''}")

        if gts:
            summary = summarize_metrics(preds, gts)
            print_eval_summary(summary)
        else:
            print("\n(No ground-truth strings found; printed JSON only.)")

        out_abs = os.path.abspath(json_path)
        with open(out_abs, "w", encoding="utf-8") as fjson:
            json.dump(json_entries, fjson, ensure_ascii=False, indent=2)
        print(f"\nJSON written to: {out_abs}")
        return out_abs


if __name__ == "__main__":
    model_ckpt = "./checkpoints/checkpoint_epoch_10.pth"
    data_dir = "/home/hpc/v123be/v123be34/UTN-CV25-Captcha-Dataset/part2/val"
    evaluator = StringEvaluator(model_ckpt)
    evaluator.evaluate(
        data_dir=data_dir,
        max_batches=None,
        json_path="part2_epoch10_test_eval_results.json",
        json_string_source="pred",
    )
