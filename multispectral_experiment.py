"""
Shared logic for the four E0-E3 multispectral YOLO11s experiments.

The four scripts in ``training_files/`` are thin entry points that call
``main(<experiment name>)``. Keeping the configuration, safeguards and test
evaluator here guarantees that all four experiments use IDENTICAL
hyperparameters, splits, metric conventions and output layout; the input
channels are the only intended difference.

This module does NOT modify the existing project modules. It reuses:
    multispectral_moudule.py            (dataset, letterbox, augmentation)
    multispectral_model.py              (YOLO11s initialisation)
    ultralytics_dataset_adapter.py      (Ultralytics batch adapter)
    multispectral_trainer.py            (custom trainer and validator)

Command-line flags understood by every training script:
    (none)         train, then evaluate best.pt on the held-out test split
    --test-only    skip training; evaluate an existing best.pt on the test split
    --preflight    quick check of safeguards + the test evaluator on a handful
                   of test images using an UNTRAINED model (no training run,
                   writes to test_results/_preflight/ only)
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from unittest import mock

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path("/home/netali/projects/Multispectral")
MODULE_DIR = PROJECT_DIR / "py_files"
TRAINING_RESULTS_DIR = PROJECT_DIR / "training_results"
TEST_RESULTS_DIR = PROJECT_DIR / "test_results"

if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import numpy as np
import torch

import multispectral_moudule as mm
from multispectral_model import initialize_multispectral_yolo
from multispectral_trainer import (
    MultispectralDetectionTrainer,
    MultispectralDetectionValidator,
)
from ultralytics_dataset_adapter import (
    MultispectralDataLoader,
    UltralyticsMultispectralDataset,
)
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER
from ultralytics.utils.metrics import ConfusionMatrix, smooth

# ---------------------------------------------------------------------------
# Fixed experiment definition
# ---------------------------------------------------------------------------

EXPECTED_EXPERIMENTS = [
    "E0_RGB",
    "E1_RGB_NIR",
    "E2_RGB_NDWI",
    "E3_RGB_NIR_NDWI",
]
EXPECTED_INPUT_CHANNELS = {
    "E0_RGB": 3,
    "E1_RGB_NIR": 4,
    "E2_RGB_NDWI": 4,
    "E3_RGB_NIR_NDWI": 5,
}
EXPECTED_CLASS_NAMES = {
    0: "visible_water",
    1: "invisible_water",
    2: "irrigated_water",
    3: "building",
}
SPLITS = ("train", "val", "test")

# ---------------------------------------------------------------------------
# Training configuration (identical for all four experiments)
# ---------------------------------------------------------------------------
#
# Everything not listed here is left at the Ultralytics 8.4.184 default.
#
# Overrides that are REQUIRED by the custom pipeline (they differ from, or do
# not exist in, the defaults):
#   model       -> local pretrained yolo11s.pt (absolute path, resolved below)
#   data        -> "multispectral": a pseudo name; the custom trainer's
#                  get_dataset() ignores it and builds the dataset from the
#                  manifests, but Ultralytics needs a non-empty value
#   task        -> "detect" (also the default)
#   project     -> training_results/ (requested output location)
#   name        -> experiment name (separate folder per experiment)
#   exist_ok    -> True so the folder is EXACTLY training_results/<name>;
#                  otherwise Ultralytics would silently create E0_RGB2, ...
#                  Accidental overwrites are prevented by our own guard
#                  (see check_output_dir_free), not by Ultralytics.
#
# Explicit but EQUAL to the Ultralytics default (written out only so the
# scientific intent is visible; verify_default_hyperparameters() proves it):
#   imgsz=640, seed=0, deterministic=True

FIXED_OVERRIDES = {
    "data": "multispectral",
    "task": "detect",
    "project": str(TRAINING_RESULTS_DIR),
    "exist_ok": True,
    "imgsz": 640,
    "seed": 0,
    "deterministic": True,
}

OVERRIDE_REASONS = {
    "model": "local pretrained yolo11s.pt for the custom model initialiser",
    "data": "pseudo name; dataset is built by the custom trainer from manifests",
    "task": "detection (same as default)",
    "project": "requested output root: training_results/",
    "name": "one folder per experiment",
    "exist_ok": "exact folder name; overwrites are blocked by our own guard",
    "imgsz": "640 (equal to the Ultralytics default)",
    "seed": "0 (equal to the Ultralytics default)",
    "deterministic": "True (equal to the Ultralytics default)",
}

# Hyperparameters that MUST equal the Ultralytics default. Checked at runtime
# against the installed version's default.yaml.
HYPERPARAMETER_KEYS = [
    "epochs", "patience", "batch", "imgsz", "optimizer", "lr0", "lrf",
    "momentum", "weight_decay", "warmup_epochs", "warmup_momentum",
    "warmup_bias_lr", "cos_lr", "close_mosaic", "amp", "box", "cls", "dfl",
    "nbs", "dropout", "multi_scale", "freeze", "fraction", "rect",
    "single_cls", "cache", "seed", "deterministic",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix",
    "copy_paste", "copy_paste_mode", "auto_augment", "erasing", "val",
    "plots", "pretrained", "profile", "compile", "time",
]

# ---------------------------------------------------------------------------
# Test-evaluation conventions (identical for all four experiments)
# ---------------------------------------------------------------------------

TEST_CONF = 0.001   # Ultralytics validation default (needed for a full PR curve / mAP)
TEST_IOU_NMS = 0.7  # Ultralytics validation NMS IoU default
TEST_MAX_DET = 300  # Ultralytics default
OPERATING_CONF = 0.25   # second confusion matrix (a realistic operating point)
CONF_MATRIX_IOU = 0.45  # Ultralytics confusion-matrix matching IoU
PREFLIGHT_IMAGES = 16

REQUIRED_GRAPHS = {
    "confusion_matrix.png": "Confusion matrix (conf 0.001, Ultralytics default)",
    "confusion_matrix_normalized.png": "Normalised confusion matrix (conf 0.001)",
    f"confusion_matrix_conf{OPERATING_CONF}.png": "Confusion matrix at conf 0.25",
    "precision_recall_curve.png": "Precision-recall curve",
    "f1_confidence_curve.png": "F1-confidence curve",
    "precision_confidence_curve.png": "Precision-confidence curve",
    "recall_confidence_curve.png": "Recall-confidence curve",
    "per_class_AP50.png": "Per-class AP@0.50",
    "per_class_AP50-95.png": "Per-class AP@0.50:0.95",
}


# ===========================================================================
# Small helpers
# ===========================================================================

def banner(text):
    line = "=" * 78
    print(f"\n{line}\n{text}\n{line}", flush=True)


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def json_safe(value):
    """Convert numpy/NaN values so json.dump produces strict JSON."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    return value


def input_channels_for(experiment):
    return sum(
        3 if channel == "rgb" else 1
        for channel in mm.EXPERIMENT_CHANNELS[experiment]
    )


def resolve_pretrained_weights():
    """Prefer a local yolo11s.pt so compute nodes do not need internet."""
    candidates = [
        MODULE_DIR / "yolo11s.pt",
        PROJECT_DIR / "yolo11s.pt",
        PROJECT_DIR / "weights" / "yolo11s.pt",
        Path.cwd() / "yolo11s.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    print(
        "WARNING: no local yolo11s.pt found in "
        f"{[str(c) for c in candidates]}; falling back to the name "
        "'yolo11s.pt', which needs internet access to download.",
        flush=True,
    )
    return "yolo11s.pt"


def check_output_dir_free(path, label):
    """Refuse to write into a non-empty directory unless explicitly allowed."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if os.environ.get("MM_ALLOW_OVERWRITE") == "1":
            print(f"WARNING: overwriting existing {label}: {path}", flush=True)
            return
        raise FileExistsError(
            f"{label} already contains files: {path}\n"
            "Refusing to overwrite. Move/rename it, or re-run with "
            "MM_ALLOW_OVERWRITE=1 if you really want to replace it."
        )


# ===========================================================================
# Safeguards: identical splits, class mapping, no test leakage
# ===========================================================================

def _manifest_path(experiment, split):
    return mm.EXPERIMENT_DIR / experiment / f"{split}.csv"


def _read_manifest(experiment, split):
    import pandas as pd

    path = _manifest_path(experiment, split)
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}. All of train.csv, val.csv and "
            "test.csv must exist for every experiment."
        )
    return pd.read_csv(path).reset_index(drop=True)


def _label_digest(label_paths):
    digest = hashlib.sha256()
    for label_path in label_paths:
        digest.update(Path(label_path).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def verify_data_integrity(experiment):
    """
    Verify, before any GPU time is spent, that:
      * the experiment/channel/class definitions are the expected ones,
      * all four experiments use the SAME images, labels and order per split,
      * train / val / test are pairwise disjoint (no test leakage),
      * every file this experiment needs exists and labels are valid.
    Returns a dict (counts + fingerprints) that is stored with the results.
    """
    if list(mm.EXPERIMENTS) != EXPECTED_EXPERIMENTS:
        raise RuntimeError(
            f"mm.EXPERIMENTS changed: {mm.EXPERIMENTS} != {EXPECTED_EXPERIMENTS}"
        )
    if dict(mm.CLASS_NAMES) != EXPECTED_CLASS_NAMES:
        raise RuntimeError(
            f"Class mapping changed: {dict(mm.CLASS_NAMES)} != "
            f"{EXPECTED_CLASS_NAMES}"
        )
    for name, expected_channels in EXPECTED_INPUT_CHANNELS.items():
        if input_channels_for(name) != expected_channels:
            raise RuntimeError(
                f"{name}: {input_channels_for(name)} input channels, "
                f"expected {expected_channels} "
                f"({mm.EXPERIMENT_CHANNELS[name]})"
            )

    manifests = {
        exp: {split: _read_manifest(exp, split) for split in SPLITS}
        for exp in mm.EXPERIMENTS
    }

    reference = mm.EXPERIMENTS[0]
    fingerprint = {"splits": {}, "class_names": EXPECTED_CLASS_NAMES}

    for split in SPLITS:
        ref_df = manifests[reference][split]
        if len(ref_df) == 0:
            raise RuntimeError(f"Split '{split}' is empty.")
        ref_ids = ref_df["image_id"].astype(str).tolist()
        if len(set(ref_ids)) != len(ref_ids):
            raise RuntimeError(f"Duplicate image_id values in split '{split}'.")

        ref_label_digest = _label_digest(ref_df["label_path"])
        for exp in mm.EXPERIMENTS[1:]:
            df = manifests[exp][split]
            if df["image_id"].astype(str).tolist() != ref_ids:
                raise RuntimeError(
                    f"Split '{split}' differs between {reference} and {exp} "
                    "(different image_ids or order)."
                )
            if df["rgb_path"].tolist() != ref_df["rgb_path"].tolist():
                raise RuntimeError(
                    f"rgb_path differs between {reference} and {exp} "
                    f"in split '{split}'."
                )
            if _label_digest(df["label_path"]) != ref_label_digest:
                raise RuntimeError(
                    f"Label file contents differ between {reference} and "
                    f"{exp} in split '{split}'."
                )

        fingerprint["splits"][split] = {
            "images": len(ref_ids),
            "image_id_sha256": sha256_text("\n".join(ref_ids)),
            "label_content_sha256": ref_label_digest,
        }

    # Pairwise disjointness of the splits (by id, RGB path and label path).
    for column in ("image_id", "rgb_path", "label_path"):
        sets = {
            split: set(manifests[reference][split][column].astype(str))
            for split in SPLITS
        }
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = sets[a] & sets[b]
            if overlap:
                raise RuntimeError(
                    f"DATA LEAKAGE: {len(overlap)} shared {column} value(s) "
                    f"between '{a}' and '{b}', e.g. {sorted(overlap)[:3]}"
                )

    # Files needed by THIS experiment exist and labels are valid.
    channels = mm.EXPERIMENT_CHANNELS[experiment]
    path_columns = ["rgb_path", "label_path"]
    if "nir" in channels:
        path_columns.append("nir_path")
    if "ndwi" in channels:
        path_columns.append("ndwi_path")

    for split in SPLITS:
        df = manifests[experiment][split]
        for column in path_columns:
            if column not in df.columns:
                raise RuntimeError(
                    f"{experiment}/{split}.csv has no column '{column}'"
                )
            missing = [p for p in df[column] if not Path(p).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} file(s) from column '{column}' missing "
                    f"in {experiment}/{split}.csv, e.g. {missing[:3]}"
                )
        for label_path in df["label_path"]:
            labels = mm.MultispectralYOLODataset.load_labels(label_path)
            if len(labels) == 0:
                continue
            class_ids = labels[:, 0]
            if not np.all(np.isin(class_ids, list(EXPECTED_CLASS_NAMES))):
                raise RuntimeError(
                    f"Unexpected class id(s) {np.unique(class_ids)} in "
                    f"{label_path}"
                )
            if labels[:, 1:].min() < 0 or labels[:, 1:].max() > 1:
                raise RuntimeError(
                    f"Normalised box values outside [0, 1] in {label_path}"
                )

    fingerprint["experiment"] = experiment
    fingerprint["channels"] = list(channels)
    fingerprint["input_channels"] = input_channels_for(experiment)
    return fingerprint


# ===========================================================================
# Training
# ===========================================================================

def build_overrides(experiment):
    overrides = dict(FIXED_OVERRIDES)
    overrides["model"] = resolve_pretrained_weights()
    overrides["name"] = experiment
    return overrides


def verify_default_hyperparameters(args):
    """Hard-fail if any listed hyperparameter differs from Ultralytics defaults."""
    effective, mismatches = {}, {}
    for key in HYPERPARAMETER_KEYS:
        if key not in DEFAULT_CFG_DICT:
            continue  # key not present in this Ultralytics version
        value = getattr(args, key)
        effective[key] = value
        if value != DEFAULT_CFG_DICT[key]:
            mismatches[key] = {"used": value, "default": DEFAULT_CFG_DICT[key]}
    if mismatches:
        raise RuntimeError(
            "Hyperparameters differ from the Ultralytics defaults: "
            f"{mismatches}"
        )
    return effective


def config_fingerprint(effective):
    return sha256_text(json.dumps(json_safe(effective), sort_keys=True))


def run_training(experiment, integrity):
    expected_dir = TRAINING_RESULTS_DIR / experiment
    check_output_dir_free(expected_dir, "training output directory")

    overrides = build_overrides(experiment)
    trainer = MultispectralDetectionTrainer(
        experiment_name=experiment, overrides=overrides
    )

    if Path(trainer.save_dir).resolve() != expected_dir.resolve():
        raise RuntimeError(
            f"Ultralytics save_dir is {trainer.save_dir}, expected "
            f"{expected_dir}. Aborting so experiments cannot mix."
        )

    effective = verify_default_hyperparameters(trainer.args)
    fingerprint = config_fingerprint(effective)

    print("Explicit overrides (everything else = Ultralytics default):")
    for key, value in overrides.items():
        print(f"  {key:14s} = {value}   # {OVERRIDE_REASONS.get(key, '')}")
    print(f"Epochs: {trainer.args.epochs} | batch: {trainer.args.batch} | "
          f"optimizer: {trainer.args.optimizer} | seed: {trainer.args.seed} | "
          f"deterministic: {trainer.args.deterministic}")
    print(f"Hyperparameter fingerprint (must match across E0-E3): "
          f"{fingerprint}", flush=True)

    trainer.train()

    best = Path(trainer.save_dir) / "weights" / "best.pt"
    last = Path(trainer.save_dir) / "weights" / "last.pt"

    summary = {
        "hyperparameters": effective,
        "hyperparameter_fingerprint": fingerprint,
        "overrides": overrides,
        "epochs_completed": int(getattr(trainer, "epoch", -1)) + 1,
        "best_fitness": _to_float(getattr(trainer, "best_fitness", None)),
    }

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best, last, summary


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return None


# ===========================================================================
# Test evaluator (custom validator on the held-out test split)
# ===========================================================================

def build_test_loader(experiment, batch_size, workers, limit=None):
    dataset = mm.MultispectralYOLODataset(
        experiment_name=experiment,
        split="test",
        image_size=640,
        augment=False,          # never augment the test split
    )
    if dataset.augment:
        raise RuntimeError("Test dataset must not use augmentation.")
    if limit is not None:
        dataset.df = dataset.df.iloc[:limit].reset_index(drop=True)

    adapter = UltralyticsMultispectralDataset(dataset)
    return MultispectralDataLoader(
        adapter,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=False,
        collate_fn=UltralyticsMultispectralDataset.collate_fn,
    )


class MultispectralTestValidator(MultispectralDetectionValidator):
    """
    Existing custom validator, pointed at the TEST split, plus:
      * canonical class names regardless of what the checkpoint stored,
      * a second confusion matrix at conf=0.25,
      * plotting errors are logged instead of aborting the evaluation.
    """

    def get_dataloader(self, dataset_path, batch_size):
        # Only used if no dataloader was passed to the constructor.
        return build_test_loader(
            self.experiment_name, batch_size, self.args.workers
        )

    def init_metrics(self, model):
        super().init_metrics(model)
        names = dict(mm.CLASS_NAMES)
        if dict(self.names) != names:
            LOGGER.warning(
                f"Checkpoint class names {dict(self.names)} replaced by "
                f"{names}."
            )
            self.names = names
            self.nc = len(names)
            self.metrics.names = names
            self.confusion_matrix = ConfusionMatrix(
                names=names, save_matches=False
            )
        self.cm_operating = ConfusionMatrix(names=names, save_matches=False)

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            self.cm_operating.process_batch(
                self._prepare_pred(pred),
                pbatch,
                conf=OPERATING_CONF,
                iou_thres=CONF_MATRIX_IOU,
            )
        super().update_metrics(preds, batch)

    def finalize_metrics(self):
        super().finalize_metrics()
        if not self.args.plots:
            return
        tmp = Path(self.save_dir) / "_cm_operating_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        for normalize in (True, False):
            self.cm_operating.plot(
                save_dir=tmp, normalize=normalize, on_plot=None
            )
        for png in tmp.glob("*.png"):
            suffix = "_normalized" if "normalized" in png.name else ""
            shutil.move(
                str(png),
                str(Path(self.save_dir)
                    / f"confusion_matrix_conf{OPERATING_CONF}{suffix}.png"),
            )
        shutil.rmtree(tmp, ignore_errors=True)

    def plot_val_samples(self, batch, *args, **kwargs):
        try:
            return super().plot_val_samples(batch, *args, **kwargs)
        except Exception as error:  # plotting must never kill the evaluation
            LOGGER.warning(f"plot_val_samples skipped: {error}")

    def plot_predictions(self, batch, *args, **kwargs):
        try:
            return super().plot_predictions(batch, *args, **kwargs)
        except Exception as error:
            LOGGER.warning(f"plot_predictions skipped: {error}")


def _fake_dataset_dict(experiment):
    names = dict(mm.CLASS_NAMES)
    return {
        "train": "multispectral_train",
        "val": "multispectral_val",
        "test": "multispectral_test",
        "nc": len(names),
        "names": names,
        "channels": input_channels_for(experiment),
    }


def run_validator_on_test(experiment, weights_path, save_dir, limit=None):
    """Run the custom validator on the test split; returns the validator."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    workers = min(
        DEFAULT_CFG_DICT["workers"],
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1),
    )
    batch_size = DEFAULT_CFG_DICT["batch"]
    loader = build_test_loader(experiment, batch_size, workers, limit=limit)

    validator_args = {
        "task": "detect",
        "mode": "val",
        "data": "multispectral",
        "split": "test",
        "imgsz": 640,
        "batch": batch_size,
        "workers": workers,
        "conf": TEST_CONF,
        "iou": TEST_IOU_NMS,
        "max_det": TEST_MAX_DET,
        "plots": True,
        "verbose": True,
        "save_json": False,
        "save_txt": False,
        "augment": False,
    }

    validator = MultispectralTestValidator(
        loader,
        save_dir=save_dir,
        args=validator_args,
        experiment_name=experiment,
    )

    # The stock standalone validation path calls check_det_dataset(<data>),
    # which would look for a dataset YAML. Provide the dataset dict ourselves.
    with mock.patch(
        "ultralytics.engine.validator.check_det_dataset",
        lambda *a, **k: _fake_dataset_dict(experiment),
    ):
        validator(model=str(weights_path))
    return validator


def _confusion_counts(matrix, names):
    """matrix is [predicted, true] with a trailing background row/col."""
    nc = len(names)
    diag = matrix.diagonal()[:nc]
    fp = matrix.sum(1)[:nc] - diag
    fn = matrix.sum(0)[:nc] - diag
    return {
        names[c]: {"tp": float(diag[c]), "fp": float(fp[c]), "fn": float(fn[c])}
        for c in range(nc)
    }


def collect_metrics(validator):
    """Extract overall/per-class metrics using the Ultralytics implementation."""
    names = dict(mm.CLASS_NAMES)
    metrics = validator.metrics
    box = metrics.box

    mp, mr, map50, map5095 = [float(x) for x in metrics.mean_results()]
    present = [int(c) for c in metrics.ap_class_index]
    nt_class = np.asarray(metrics.nt_per_class)
    nt_image = np.asarray(metrics.nt_per_image)
    eps = 1e-16

    per_class = []
    for c in sorted(names):
        n_inst = int(nt_class[c]) if c < len(nt_class) else 0
        n_img = int(nt_image[c]) if c < len(nt_image) else 0
        row = {
            "class_id": c, "class": names[c],
            "images": n_img, "instances": n_inst,
            "precision": math.nan, "recall": math.nan, "f1": math.nan,
            "map50": math.nan, "map50_95": math.nan,
            "tp": math.nan, "fp": math.nan, "fn": math.nan,
        }
        if c in present:
            i = present.index(c)
            p, r, f1 = float(box.p[i]), float(box.r[i]), float(box.f1[i])
            tp = float(np.round(r * n_inst))
            fp = float(np.round(tp / (p + eps) - tp))
            row.update(
                precision=p, recall=r, f1=f1,
                map50=float(box.all_ap[i, 0]),
                map50_95=float(box.all_ap[i].mean()),
                tp=tp, fp=fp, fn=float(n_inst - tp),
            )
        per_class.append(row)

    valid = [r for r in per_class if not math.isnan(r["f1"])]
    overall = {
        "class": "all",
        "images": int(validator.seen),
        "instances": int(sum(r["instances"] for r in per_class)),
        "precision": mp,
        "recall": mr,
        "f1": float(np.mean([r["f1"] for r in valid])) if valid else math.nan,
        "map50": map50,
        "map50_95": map5095,
        "tp": float(sum(r["tp"] for r in valid)),
        "fp": float(sum(r["fp"] for r in valid)),
        "fn": float(sum(r["fn"] for r in valid)),
    }

    conf_max_f1 = math.nan
    try:
        f1_curve = np.asarray(box.f1_curve)
        px = np.asarray(box.px)
        conf_max_f1 = float(px[smooth(f1_curve.mean(0), 0.1).argmax()])
    except Exception as error:
        LOGGER.warning(f"Could not determine max-F1 confidence: {error}")

    bg = list(names.values()) + ["background"]
    cm_default = np.asarray(validator.confusion_matrix.matrix)
    cm_operating = np.asarray(validator.cm_operating.matrix)

    return {
        "overall": overall,
        "per_class": per_class,
        "operating_point_conf_max_mean_f1": conf_max_f1,
        "confusion_matrix_labels": bg,
        "confusion_matrix_conf_0.001": cm_default,
        f"confusion_matrix_conf_{OPERATING_CONF}": cm_operating,
        "confusion_counts_conf_0.25": _confusion_counts(cm_operating, names),
        "speed_ms_per_image": dict(validator.speed),
    }


CSV_COLUMNS = [
    "class", "class_id", "images", "instances", "precision", "recall", "f1",
    "map50", "map50_95", "tp", "fp", "fn",
]


def _write_outputs(experiment, out_dir, results, provenance, preflight=False):
    out_dir = Path(out_dir)
    overall, per_class = results["overall"], results["per_class"]

    # ---- CSV ------------------------------------------------------------
    with open(out_dir / "test_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in [{**overall, "class_id": ""}] + per_class:
            writer.writerow({
                k: ("" if isinstance(row.get(k), float) and math.isnan(row[k])
                    else row.get(k, ""))
                for k in CSV_COLUMNS
            })

    # ---- confusion matrices as CSV ---------------------------------------
    labels = results["confusion_matrix_labels"]
    for key in ("confusion_matrix_conf_0.001",
                f"confusion_matrix_conf_{OPERATING_CONF}"):
        with open(out_dir / f"{key}.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["predicted \\ true"] + labels)
            for label, row in zip(labels, results[key]):
                writer.writerow([label] + [int(v) for v in row])

    # ---- JSON -------------------------------------------------------------
    payload = {
        "experiment": experiment,
        "split": "test",
        "preflight_untrained_model": preflight,
        "conventions": {
            "map_conf_threshold": TEST_CONF,
            "nms_iou": TEST_IOU_NMS,
            "max_det": TEST_MAX_DET,
            "map_iou_thresholds": "0.50:0.05:0.95 (10 thresholds)",
            "precision_recall_f1_operating_point":
                "confidence maximising the smoothed mean-over-classes F1 "
                "curve (Ultralytics convention); chosen on the test "
                "predictions themselves",
            "overall_precision_recall_map":
                "unweighted mean over classes that have test instances",
            "overall_f1": "unweighted mean of per-class F1 at that confidence",
            "tp_fp_fn": "per class at the same operating point "
                        "(tp=round(recall*n), fp=round(tp/precision-tp), "
                        "fn=n-tp); overall = sum over classes",
            "confusion_matrix_iou_match": CONF_MATRIX_IOU,
            "confusion_matrix_confidences": [TEST_CONF, OPERATING_CONF],
            "confusion_matrix_layout": "rows=predicted, cols=true, "
                                       "last row/col = background",
            "tta": False,
            "augmentation": False,
        },
        "operating_point_confidence": results["operating_point_conf_max_mean_f1"],
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix_labels": labels,
        "confusion_matrix_conf_0.001": results["confusion_matrix_conf_0.001"],
        f"confusion_matrix_conf_{OPERATING_CONF}":
            results[f"confusion_matrix_conf_{OPERATING_CONF}"],
        "confusion_counts_conf_0.25": results["confusion_counts_conf_0.25"],
        "speed_ms_per_image": results["speed_ms_per_image"],
        "provenance": provenance,
    }
    with open(out_dir / "test_metrics.json", "w") as handle:
        json.dump(json_safe(payload), handle, indent=2)

    # ---- per-class AP bar charts -------------------------------------------
    _plot_per_class_ap(out_dir, experiment, per_class)

    # ---- renamed copies of the Ultralytics curves ---------------------------
    for source, target in (
        ("BoxPR_curve.png", "precision_recall_curve.png"),
        ("BoxF1_curve.png", "f1_confidence_curve.png"),
        ("BoxP_curve.png", "precision_confidence_curve.png"),
        ("BoxR_curve.png", "recall_confidence_curve.png"),
    ):
        if (out_dir / source).is_file():
            shutil.copy2(out_dir / source, out_dir / target)

    # ---- text summary --------------------------------------------------------
    lines = [
        f"{'PREFLIGHT (untrained model) - ' if preflight else ''}"
        f"TEST RESULTS: {experiment}",
        f"Channels: {provenance.get('channels')} "
        f"({provenance.get('input_channels')} input channels)",
        f"Checkpoint: {provenance.get('checkpoint')}",
        f"Test images: {overall['images']} | instances: {overall['instances']}",
        f"Operating confidence (max mean F1): "
        f"{results['operating_point_conf_max_mean_f1']:.4f}",
        "",
        f"{'class':18s}{'P':>8s}{'R':>8s}{'F1':>8s}{'mAP50':>8s}"
        f"{'mAP50-95':>10s}{'TP':>7s}{'FP':>7s}{'FN':>7s}{'n':>7s}",
    ]
    for row in [overall] + per_class:
        def fmt(key, width=8, digits=4):
            value = row[key]
            return (f"{'n/a':>{width}s}" if isinstance(value, float)
                    and math.isnan(value) else f"{value:>{width}.{digits}f}")
        def cnt(key):
            value = row[key]
            return (f"{'n/a':>7s}" if isinstance(value, float)
                    and math.isnan(value) else f"{int(value):>7d}")
        lines.append(
            f"{row['class']:18s}{fmt('precision')}{fmt('recall')}{fmt('f1')}"
            f"{fmt('map50')}{fmt('map50_95', 10)}"
            f"{cnt('tp')}{cnt('fp')}{cnt('fn')}{int(row['instances']):>7d}"
        )
    lines += [
        "",
        f"Conventions: mAP at conf>={TEST_CONF}, NMS IoU {TEST_IOU_NMS}, "
        "IoU 0.50:0.95; P/R/F1/TP/FP/FN at the max-mean-F1 confidence; "
        "overall values are unweighted class means.",
    ]
    (out_dir / "test_summary.txt").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def _plot_per_class_ap(out_dir, experiment, per_class):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for key, title, filename in (
        ("map50", "AP@0.50", "per_class_AP50.png"),
        ("map50_95", "AP@0.50:0.95", "per_class_AP50-95.png"),
    ):
        names = [r["class"] for r in per_class]
        values = [0.0 if math.isnan(r[key]) else r[key] for r in per_class]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(names, values)
        for bar, row in zip(bars, per_class):
            label = ("no test\ninstances" if math.isnan(row[key])
                     else f"{row[key]:.3f}")
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    label, ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(title)
        ax.set_title(f"{experiment} - per-class {title} (test split)")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(Path(out_dir) / filename, dpi=200)
        plt.close(fig)


def run_test_evaluation(experiment, best_pt, out_dir, integrity,
                        train_summary=None, limit=None, preflight=False):
    best_pt = Path(best_pt)
    if not best_pt.is_file():
        raise FileNotFoundError(f"best checkpoint not found: {best_pt}")
    out_dir = Path(out_dir)
    check_output_dir_free(out_dir, "test output directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    sha_before = sha256_file(best_pt)

    ckpt_info = {}
    try:
        ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
        ckpt_info = {
            "epoch_saved": ckpt.get("epoch"),
            "best_fitness": _to_float(ckpt.get("best_fitness")),
        }
        del ckpt
    except Exception as error:
        ckpt_info = {"note": f"could not read checkpoint metadata: {error}"}

    started = time.time()
    validator = run_validator_on_test(experiment, best_pt, out_dir, limit=limit)
    results = collect_metrics(validator)

    sha_after = sha256_file(best_pt)
    if sha_before != sha_after:
        raise RuntimeError("best.pt changed during test evaluation!")

    provenance = {
        "checkpoint": str(best_pt),
        "checkpoint_sha256": sha_after,
        "checkpoint_info": ckpt_info,
        "channels": list(mm.EXPERIMENT_CHANNELS[experiment]),
        "input_channels": input_channels_for(experiment),
        "data_integrity": integrity,
        "training": train_summary,
        "evaluated_images": results["overall"]["images"],
        "image_limit": limit,
        "evaluation_seconds": round(time.time() - started, 1),
        "timestamp": datetime.now().astimezone().isoformat(),
        "ultralytics_version": _ultralytics_version(),
        "torch_version": torch.__version__,
    }
    text = _write_outputs(experiment, out_dir, results, provenance, preflight)

    missing = [name for name in REQUIRED_GRAPHS if not (out_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing expected test graphs: {missing}")
    return results, text


def _ultralytics_version():
    import ultralytics
    return ultralytics.__version__


# ===========================================================================
# Preflight (untrained model, a few test images, separate output folder)
# ===========================================================================

def run_preflight(experiment, integrity):
    out_dir = TEST_RESULTS_DIR / "_preflight" / experiment
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    model = initialize_multispectral_yolo(
        weights=resolve_pretrained_weights(),
        input_channels=input_channels_for(experiment),
        num_classes=len(mm.CLASS_NAMES),
        device=None,
    )
    model.names = dict(mm.CLASS_NAMES)
    ckpt_path = out_dir / "untrained_preflight.pt"
    torch.save(
        {
            "epoch": -1, "best_fitness": None, "model": None,
            "ema": deepcopy(model).half(), "train_args": {},
        },
        ckpt_path,
    )
    del model

    eval_dir = out_dir / "eval"
    results, text = run_test_evaluation(
        experiment, ckpt_path, eval_dir, integrity,
        train_summary=None, limit=PREFLIGHT_IMAGES, preflight=True,
    )
    print(text)
    print(f"\nPREFLIGHT PASSED for {experiment}: evaluator ran on "
          f"{results['overall']['images']} test images; outputs in {eval_dir} "
          "(metrics are meaningless: untrained detection head).", flush=True)


# ===========================================================================
# Entry point
# ===========================================================================

def print_header(experiment, integrity):
    banner(f"EXPERIMENT {experiment}")
    print(f"Channels          : {mm.EXPERIMENT_CHANNELS[experiment]}")
    print(f"Input channels    : {input_channels_for(experiment)}")
    print(f"Classes           : {dict(mm.CLASS_NAMES)}")
    for split in SPLITS:
        info = integrity["splits"][split]
        print(f"{split:5s} images     : {info['images']}   "
              f"(ids sha256 {info['image_id_sha256'][:12]})")
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | "
          f"ultralytics {_ultralytics_version()} | "
          f"CUDA {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})"
             if torch.cuda.is_available() else ""), flush=True)


def main(experiment, argv=None):
    parser = argparse.ArgumentParser(description=f"Train and test {experiment}")
    parser.add_argument("--test-only", action="store_true",
                        help="skip training, evaluate existing best.pt")
    parser.add_argument("--preflight", action="store_true",
                        help="check safeguards + test evaluator on a few "
                             "test images with an untrained model")
    flags = parser.parse_args(argv)

    try:
        if experiment not in mm.EXPERIMENTS:
            raise ValueError(f"Unknown experiment {experiment}")

        integrity = verify_data_integrity(experiment)
        print_header(experiment, integrity)

        if flags.preflight:
            run_preflight(experiment, integrity)
            return 0

        train_dir = TRAINING_RESULTS_DIR / experiment
        test_dir = TEST_RESULTS_DIR / experiment
        check_output_dir_free(test_dir, "test output directory")

        if flags.test_only:
            best = train_dir / "weights" / "best.pt"
            train_summary = {"note": "evaluation only (--test-only)"}
        else:
            banner(f"TRAINING {experiment}")
            best, _last, train_summary = run_training(experiment, integrity)

        if not best.is_file():
            raise FileNotFoundError(
                f"Training did not produce best.pt: {best}"
            )

        banner(f"TEST EVALUATION {experiment} (held-out test split)")
        results, text = run_test_evaluation(
            experiment, best, test_dir, integrity, train_summary
        )

        banner(f"FINAL SUMMARY {experiment}")
        print(text)
        print(f"\nTraining outputs : {train_dir}")
        print(f"Best checkpoint  : {best}")
        print(f"Test outputs     : {test_dir}")
        print(f"\n=== {experiment} COMPLETED SUCCESSFULLY ===", flush=True)
        return 0

    except Exception as error:
        print(f"\n!!! {experiment} FAILED: {type(error).__name__}: {error}",
              file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1