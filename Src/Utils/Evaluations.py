# Src/Utils/Evaluations.py
from pathlib import Path

def levenshtein(a: str, b: str) -> int:
    """
    Compute the Levenshtein distance between two strings.

    Args:
        a (str): First string.
        b (str): Second string.

    Returns:
        int: Minimum number of edits to transform `a` into `b`.
    """
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def compute_accuracy(preds, gts) -> float:
    """
    Compute exact-match accuracy.

    Args:
        preds (List[str]): Predicted strings.
        gts (List[str]): Ground-truth strings.

    Returns:
        float: Fraction of predictions exactly matching ground truth.
    """
    n = max(1, len(gts))
    return sum(p == g for p, g in zip(preds, gts)) / n


def compute_ler_and_totals(preds, gts):
    """
    Compute Levenshtein Error Rate and supporting totals.

    Args:
        preds (List[str]): Predicted strings.
        gts (List[str]): Ground-truth strings.

    Returns:
        Tuple[float, int, int]: (LER, total_lev, total_gt_len).
    """
    total_dist, total_len = 0, 0
    for p, g in zip(preds, gts):
        total_dist += levenshtein(p, g)
        total_len += max(1, len(g))
    ler = total_dist / total_len if total_len > 0 else 0.0
    return ler, total_dist, total_len


def summarize_metrics(preds, gts):
    """
    Aggregate core evaluation metrics.

    Args:
        preds (List[str]): Predicted strings.
        gts (List[str]): Ground-truth strings.

    Returns:
        dict: Summary with samples, accuracy, LER, total_lev, total_gt_len, and lev_div_20000.
    """
    acc = compute_accuracy(preds, gts) if gts else 0.0
    ler, total_lev, total_len = compute_ler_and_totals(preds, gts) if gts else (0.0, 0, 0)
    return {
        "samples": len(gts),
        "accuracy": acc,
        "ler": ler,
        "total_lev": total_lev,
        "total_gt_len": total_len,
        "lev_div_20000": total_lev / 20000.0 if total_lev else 0.0,
    }


def print_eval_summary(summary: dict) -> None:
    """
    Print a concise evaluation summary.

    Args:
        summary (dict): Metrics dictionary from `summarize_metrics`.
    """
    print("\n----- Summary -----")
    print(f"Samples:              {summary.get('samples', 0)}")
    print(f"Accuracy:             {summary.get('accuracy', 0.0)*100:.2f}%")
    print(f"LER:                  {summary.get('ler', 0.0):.4f}")
    print(f"Levenshtein/20000:    {summary.get('lev_div_20000', 0.0):.6f}  "
          f"(total_lev={summary.get('total_lev', 0)}, total_gt_len={summary.get('total_gt_len', 0)})")


def get_idx_to_char(num_classes: int) -> dict:
    """
    Build a mapping from class indices to characters (0–9, A–Z).
    Assumes the last index is reserved for the CTC blank.

    Args:
        num_classes (int): Total number of output classes, including blank.

    Returns:
        dict: Mapping {index: character}.
    """
    # Ensure at least enough classes for 0-9 + A-Z + blank
    printable_size = min(36, num_classes - 1)

    idx_to_char = {i: str(i) for i in range(10)}  # 0-9
    idx_to_char.update({10 + i: chr(ord("A") + i) for i in range(26)})  # A-Z

    # Trim if num_classes < 37 (edge cases)
    idx_to_char = {k: v for k, v in idx_to_char.items() if k < printable_size}

    return idx_to_char


def get_char_to_cat() -> dict:
    """
    Build a mapping from characters (0–9, A–Z) to category IDs (0–35).

    Returns:
        dict: Mapping {character: category_id}.
    """
    char_to_cat = {str(i): i for i in range(10)}  # 0-9 → 0–9
    char_to_cat.update({chr(ord("A") + i): 10 + i for i in range(26)})  # A-Z → 10–35
    return char_to_cat

from pathlib import Path

def build_json_entry(
    height: int,
    width: int,
    image_path: str,
    captcha_string: str,
    char_to_cat: dict,
    index: int = None,
) -> dict:
    """
    Build a JSON entry in the dataset-style format used for evaluation.

    Args:
        height (int): Image height.
        width (int): Image width.
        image_path (str): Path to the image file (used to derive image_id).
        captcha_string (str): Predicted or ground-truth CAPTCHA string.
        char_to_cat (dict): Mapping {character: category_id}.
        index (int, optional): Fallback index if image_path is missing.

    Returns:
        dict: JSON entry containing height, width, image_id, captcha_string,
              and annotations (with placeholder boxes + category IDs).
    """
    # Derive image_id from path or fallback index
    if image_path:
        image_id = Path(image_path).stem
    else:
        image_id = f"{index+1:06d}" if index is not None else "unknown"

    annotations = []
    for ch in captcha_string:
        k = char_to_cat.get(ch.upper(), -1)
        if k == -1:
            continue
        annotations.append({
            "bbox": [0, 0, 0, 0],               # placeholder
            "oriented_bbox": [0, 0, 0, 0, 0, 0, 0, 0],  # placeholder
            "category_id": k,
        })

    return {
        "height": int(height),
        "width": int(width),
        "image_id": image_id,
        "captcha_string": captcha_string,
        "annotations": annotations,
    }

def levenshtein(a: str, b: str) -> int:
    """
    Compute the Levenshtein distance between two strings.

    Args:
        a (str): First string.
        b (str): Second string.

    Returns:
        int: Minimum number of edits to transform `a` into `b`.
    """
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def compute_accuracy(preds, gts) -> float:
    """
    Compute exact-match accuracy.

    Args:
        preds (List[str]): Predicted strings.
        gts (List[str]): Ground-truth strings.

    Returns:
        float: Fraction of predictions exactly matching ground truth.
    """
    n = max(1, len(gts))
    return sum(p == g for p, g in zip(preds, gts)) / n


def compute_ler_and_totals(preds, gts):
    """
    Compute Levenshtein Error Rate and supporting totals.

    Args:
        preds (List[str]): Predicted strings.
        gts (List[str]): Ground-truth strings.

    Returns:
        Tuple[float, int, int]: (LER, total_lev, total_gt_len).
    """
    total_dist, total_len = 0, 0
    for p, g in zip(preds, gts):
        total_dist += levenshtein(p, g)
        total_len += max(1, len(g))
    ler = total_dist / total_len if total_len > 0 else 0.0
    return ler, total_dist, total_len


def summarize_metrics(preds, gts):
    """
    Aggregate core evaluation metrics.

    Args:
        preds (List[str]): Predicted strings.
        gts (List[str]): Ground-truth strings.

    Returns:
        dict: Summary with samples, accuracy, LER, total_lev, total_gt_len, and lev_div_20000.
    """
    acc = compute_accuracy(preds, gts) if gts else 0.0
    ler, total_lev, total_len = compute_ler_and_totals(preds, gts) if gts else (0.0, 0, 0)
    return {
        "samples": len(gts),
        "accuracy": acc,
        "ler": ler,
        "total_lev": total_lev,
        "total_gt_len": total_len,
        "lev_div_20000": total_lev / 20000.0 if total_lev else 0.0,
    }


def print_eval_summary(summary: dict) -> None:
    """
    Print a concise evaluation summary.

    Args:
        summary (dict): Metrics dictionary from `summarize_metrics`.
    """
    print("\n----- Summary -----")
    print(f"Samples:              {summary.get('samples', 0)}")
    print(f"Accuracy:             {summary.get('accuracy', 0.0)*100:.2f}%")
    print(f"LER:                  {summary.get('ler', 0.0):.4f}")
    print(f"Levenshtein/20000:    {summary.get('lev_div_20000', 0.0):.6f}  "
          f"(total_lev={summary.get('total_lev', 0)}, total_gt_len={summary.get('total_gt_len', 0)})")


def get_idx_to_char(num_classes: int) -> dict:
    """
    Build a mapping from class indices to characters (0–9, A–Z).
    Assumes the last index is reserved for the CTC blank.

    Args:
        num_classes (int): Total number of output classes, including blank.

    Returns:
        dict: Mapping {index: character}.
    """
    # Ensure at least enough classes for 0-9 + A-Z + blank
    printable_size = min(36, num_classes - 1)

    idx_to_char = {i: str(i) for i in range(10)}  # 0-9
    idx_to_char.update({10 + i: chr(ord("A") + i) for i in range(26)})  # A-Z

    # Trim if num_classes < 37 (edge cases)
    idx_to_char = {k: v for k, v in idx_to_char.items() if k < printable_size}

    return idx_to_char


def get_char_to_cat() -> dict:
    """
    Build a mapping from characters (0–9, A–Z) to category IDs (0–35).

    Returns:
        dict: Mapping {character: category_id}.
    """
    char_to_cat = {str(i): i for i in range(10)}  # 0-9 → 0–9
    char_to_cat.update({chr(ord("A") + i): 10 + i for i in range(26)})  # A-Z → 10–35
    return char_to_cat

from pathlib import Path

def build_json_entry(
    height: int,
    width: int,
    image_path: str,
    captcha_string: str,
    char_to_cat: dict,
    index: int = None,
) -> dict:
    """
    Build a JSON entry in the dataset-style format used for evaluation.

    Args:
        height (int): Image height.
        width (int): Image width.
        image_path (str): Path to the image file (used to derive image_id).
        captcha_string (str): Predicted or ground-truth CAPTCHA string.
        char_to_cat (dict): Mapping {character: category_id}.
        index (int, optional): Fallback index if image_path is missing.

    Returns:
        dict: JSON entry containing height, width, image_id, captcha_string,
              and annotations (with placeholder boxes + category IDs).
    """
    # Derive image_id from path or fallback index
    if image_path:
        image_id = Path(image_path).stem
    else:
        image_id = f"{index+1:06d}" if index is not None else "unknown"

    annotations = []
    for ch in captcha_string:
        k = char_to_cat.get(ch.upper(), -1)
        if k == -1:
            continue
        annotations.append({
            "bbox": [0, 0, 0, 0],               # placeholder
            "oriented_bbox": [0, 0, 0, 0, 0, 0, 0, 0],  # placeholder
            "category_id": k,
        })

    return {
        "height": int(height),
        "width": int(width),
        "image_id": image_id,
        "captcha_string": captcha_string,
        "annotations": annotations,
    }
