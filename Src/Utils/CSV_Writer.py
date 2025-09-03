# csv_logger.py
import os, csv
from datetime import datetime

class CSVLogger:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.path_batches = os.path.join(save_dir, "train_batches.csv")
        self.path_epochs  = os.path.join(save_dir, "train_epochs.csv")
        self.path_val     = os.path.join(save_dir, "val_epochs.csv")

        # create empty files so paths exist
        for p in (self.path_batches, self.path_epochs, self.path_val):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            if not os.path.exists(p):
                open(p, "a").close()

        # internal state
        self._batch_header_written = False
        self._epoch_header_written = False
        self._val_header_written   = False
        self._batch_header = None
        self._epoch_header = None

    @staticmethod
    def _utc_now():
        return datetime.utcnow().isoformat(timespec="seconds")

    @staticmethod
    def _append(path, row):
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    @staticmethod
    def _ensure_header(path, header):
        need = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
        if need:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    # ---------- batches ----------
    def init_batch_header(self, component_keys):
        """component_keys: e.g. ['total_loss','bbox_loss','obj_loss','class_loss']"""
        others = [k for k in component_keys if k != 'total_loss']
        self._batch_header = ["timestamp","epoch","batch_idx","num_batches","lr","total_loss"] \
                             + others + ["avg_total_loss"]
        self._ensure_header(self.path_batches, self._batch_header)
        self._batch_header_written = True

    def log_batch(self, *, epoch, batch_idx, num_batches, lr, loss_dict, running_avg_total):
        """loss_dict must at least have 'total_loss' and any component keys used in init_batch_header()."""
        if not self._batch_header_written:
            # derive header order from current dict if not initialized
            comp_keys = [k for k in loss_dict.keys() if (k == "total_loss" or k.endswith("_loss"))]
            # prefer a canonical order
            preferred = ["total_loss","bbox_loss","obj_loss","class_loss"]
            ordered = [k for k in preferred if k in comp_keys] + [k for k in comp_keys if k not in preferred]
            self.init_batch_header(ordered)

        row = [
            self._utc_now(),
            epoch,
            batch_idx,
            num_batches,
            f"{lr:.8f}",
            f"{float(loss_dict['total_loss']):.6f}",
        ]
        # components between total and avg_total_loss, respect header order
        for c in self._batch_header[6:-1]:
            val = float(loss_dict.get(c, 0.0))
            row.append(f"{val:.6f}")
        row.append(f"{running_avg_total:.6f}")
        self._append(self.path_batches, row)

    # ---------- epochs ----------
    def init_epoch_header(self, component_keys):
        others = [k for k in component_keys if k != 'total_loss']
        self._epoch_header = ["timestamp","epoch","lr","is_best","avg_total_loss"] \
                             + [f"avg_{k}" for k in others]
        self._ensure_header(self.path_epochs, self._epoch_header)
        self._epoch_header_written = True

    def log_epoch(self, *, epoch, lr, is_best, avg_total_loss, avg_components: dict):
        """avg_components: dict of {component_name: avg_value} excluding 'total_loss'."""
        if not self._epoch_header_written:
            keys = ["total_loss"] + list(avg_components.keys())
            self.init_epoch_header(keys)

        row = [
            self._utc_now(),
            epoch,
            f"{lr:.8f}",
            int(bool(is_best)),
            f"{avg_total_loss:.6f}",
        ]
        for hdr in self._epoch_header[5:]:
            k = hdr.replace("avg_", "")
            row.append(f"{float(avg_components.get(k, 0.0)):.6f}")
        self._append(self.path_epochs, row)

    # ---------- validation ----------
    def log_val(self, *, epoch, avg_val_loss, lr):
        header = ["timestamp","epoch","avg_val_loss","lr"]
        if not self._val_header_written:
            self._ensure_header(self.path_val, header)
            self._val_header_written = True
        self._append(self.path_val, [
            self._utc_now(),
            epoch,
            f"{avg_val_loss:.6f}",
            f"{lr:.8f}",
        ])
