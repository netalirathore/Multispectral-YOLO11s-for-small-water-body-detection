"""
Multispectral YOLO dataset utilities for the E0-E3 experiments.

Import this module from each experiment's training script.

The augmentation seed is derived from:
    global_seed + epoch + image_id

Therefore, the same image receives the same geometric transform
across E0_RGB, E1_RGB_NIR, E2_RGB_NDWI and E3_RGB_NIR_NDWI
for a given epoch, while transforms can vary between epochs.
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE = Path(
    "/home/netali/projects/Multispectral/Prepared_Multispectral"
)
EXPERIMENT_DIR = BASE / "experiments"

IMAGE_SIZE = 640
GLOBAL_AUGMENTATION_SEED = 42

EXPERIMENTS = [
    "E0_RGB",
    "E1_RGB_NIR",
    "E2_RGB_NDWI",
    "E3_RGB_NIR_NDWI",
]

EXPERIMENT_CHANNELS = {
    "E0_RGB": ["rgb"],
    "E1_RGB_NIR": ["rgb", "nir"],
    "E2_RGB_NDWI": ["rgb", "ndwi"],
    "E3_RGB_NIR_NDWI": ["rgb", "nir", "ndwi"],
}

CLASS_NAMES = {
    0: "visible_water",
    1: "invisible_water",
    2: "irrigated_water",
    3: "building",
}


# ============================================================
# 2. EPOCH-DEPENDENT SYNCHRONIZED AUGMENTATION
# ============================================================

def get_augmentation_params(
    image_id,
    epoch,
    horizontal_flip=0.5,
    vertical_flip=0.5,
    rotate90=True,
    rotation_probability=0.5,
):
    """Return deterministic geometric augmentation parameters."""

    for name, probability in (
        ("horizontal_flip", horizontal_flip),
        ("vertical_flip", vertical_flip),
        ("rotation_probability", rotation_probability),
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    seed_string = (
        f"{GLOBAL_AUGMENTATION_SEED}|{int(epoch)}|{str(image_id)}"
    )
    seed = int(
        hashlib.sha256(seed_string.encode()).hexdigest()[:8],
        16,
    )
    rng = np.random.default_rng(seed)

    params = {
        "horizontal_flip": rng.random() < horizontal_flip,
        "vertical_flip": rng.random() < vertical_flip,
        "rotate": bool(rotate90 and rng.random() < rotation_probability),
        "rotation_k": 0,
    }

    if params["rotate"]:
        params["rotation_k"] = int(rng.choice([1, 2, 3]))

    return params


# ============================================================
# 3. LETTERBOX AND LABEL TRANSFORMATION
# ============================================================

def letterbox_multispectral(image, target_size=640, pad_values=None):
    """
    Resize an HWC multispectral image while preserving aspect ratio.

    RGB/NIR padding should be 114/255; NDWI padding should be 0.
    Returns image, scale, dw, dh.
    """

    h, w, channels = image.shape
    scale = min(target_size / h, target_size / w)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    if pad_values is None:
        pad_values = [0.0] * channels

    pad_values = np.asarray(pad_values, dtype=np.float32)
    if len(pad_values) != channels:
        raise ValueError(
            f"Expected {channels} pad values, got {len(pad_values)}"
        )

    output = np.empty(
        (target_size, target_size, channels),
        dtype=np.float32,
    )

    for channel_index in range(channels):
        output[:, :, channel_index] = pad_values[channel_index]

    dw = (target_size - new_w) / 2
    dh = (target_size - new_h) / 2

    left = int(round(dw - 0.1))
    top = int(round(dh - 0.1))

    output[
        top:top + new_h,
        left:left + new_w,
        :
    ] = resized

    return output, scale, dw, dh


def transform_labels_for_letterbox(
    labels,
    original_width,
    original_height,
    scale,
    dw,
    dh,
    target_size,
):
    """Transform normalized YOLO labels to letterboxed coordinates."""

    if len(labels) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    labels = labels.copy()

    labels[:, 1] = (
        labels[:, 1] * original_width * scale + dw
    ) / target_size

    labels[:, 2] = (
        labels[:, 2] * original_height * scale + dh
    ) / target_size

    labels[:, 3] = (
        labels[:, 3] * original_width * scale
    ) / target_size

    labels[:, 4] = (
        labels[:, 4] * original_height * scale
    ) / target_size

    return labels.astype(np.float32)


# ============================================================
# 4. APPLY GEOMETRIC AUGMENTATION TO IMAGE AND BOXES
# ============================================================

def apply_augmentation(image, labels, params):
    """Apply identical flips/90-degree rotation to every channel and box."""

    image = image.copy()
    labels = labels.copy()

    if params["horizontal_flip"]:
        image = np.fliplr(image).copy()
        if len(labels) > 0:
            labels[:, 1] = 1.0 - labels[:, 1]

    if params["vertical_flip"]:
        image = np.flipud(image).copy()
        if len(labels) > 0:
            labels[:, 2] = 1.0 - labels[:, 2]

    if params["rotate"]:
        k = params["rotation_k"]
        image = np.rot90(image, k=k, axes=(0, 1)).copy()

        if len(labels) > 0:
            x = labels[:, 1].copy()
            y = labels[:, 2].copy()
            w = labels[:, 3].copy()
            h = labels[:, 4].copy()

            if k == 1:
                labels[:, 1] = y
                labels[:, 2] = 1.0 - x
                labels[:, 3] = h
                labels[:, 4] = w

            elif k == 2:
                labels[:, 1] = 1.0 - x
                labels[:, 2] = 1.0 - y

            elif k == 3:
                labels[:, 1] = 1.0 - y
                labels[:, 2] = x
                labels[:, 3] = h
                labels[:, 4] = w

    return image, labels


# ============================================================
# 5. MULTISPECTRAL DATASET
# ============================================================

class MultispectralYOLODataset(Dataset):
    """Custom dataset for RGB, RGB+NIR, RGB+NDWI and RGB+NIR+NDWI."""

    def __init__(
        self,
        experiment_name,
        split="train",
        image_size=640,
        augment=False,
        horizontal_flip=0.5,
        vertical_flip=0.5,
        rotate90=True,
        rotation_probability=0.5,
    ):
        if experiment_name not in EXPERIMENT_CHANNELS:
            raise ValueError(
                f"Unknown experiment: {experiment_name}. "
                f"Choose from {list(EXPERIMENT_CHANNELS)}"
            )

        self.experiment_name = experiment_name
        self.split = split
        self.image_size = image_size
        self.channels = EXPERIMENT_CHANNELS[experiment_name]
        self.augment = bool(augment and split == "train")

        # Shared-memory scalar is visible to DataLoader worker processes,
        # including persistent workers.
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()

        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotate90 = rotate90
        self.rotation_probability = rotation_probability

        for name, probability in (
            ("horizontal_flip", horizontal_flip),
            ("vertical_flip", vertical_flip),
            ("rotation_probability", rotation_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        manifest_path = (
            EXPERIMENT_DIR / experiment_name / f"{split}.csv"
        )
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}"
            )

        self.df = pd.read_csv(manifest_path)

        required_columns = ["image_id", "rgb_path", "label_path"]
        if "nir" in self.channels:
            required_columns.append("nir_path")
        if "ndwi" in self.channels:
            required_columns.append("ndwi_path")

        missing_columns = [
            column for column in required_columns
            if column not in self.df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing manifest columns for {experiment_name}: "
                f"{missing_columns}"
            )

        self.df = self.df.reset_index(drop=True)

        print(
            f"{experiment_name} | {split}: {len(self.df)} images | "
            f"Channels: {self.channels} | Augmentation: {self.augment}"
        )

    def __len__(self):
        return len(self.df)

    def set_epoch(self, epoch):
        """Update the epoch used to derive deterministic augmentation."""
        self.epoch.fill_(int(epoch))

    @staticmethod
    def load_rgb(path):
        image = np.array(
            Image.open(path).convert("RGB"),
            dtype=np.float32,
        )
        return image / 255.0

    @staticmethod
    def load_nir(path):
        image = np.array(
            Image.open(path).convert("RGB"),
            dtype=np.float32,
        )
        # NIR channels are identical in the verified train/validation pool.
        nir = image[:, :, 0]
        return nir[:, :, np.newaxis] / 255.0

    @staticmethod
    def load_ndwi(path):
        ndwi = np.load(path).astype(np.float32)
        return ndwi[:, :, np.newaxis]

    @staticmethod
    def load_labels(path):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Label file not found: {path}")

        if path.stat().st_size == 0:
            return np.zeros((0, 5), dtype=np.float32)

        labels = np.loadtxt(path, dtype=np.float32, ndmin=2)

        if labels.shape[1] != 5:
            raise ValueError(
                f"Expected 5 label columns in {path}, "
                f"found {labels.shape[1]}"
            )

        return labels

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_id = str(row["image_id"])

        channel_arrays = []
        pad_values = []

        if "rgb" in self.channels:
            rgb = self.load_rgb(row["rgb_path"])
            channel_arrays.append(rgb)
            pad_values.extend([114.0 / 255.0] * 3)

        if "nir" in self.channels:
            nir = self.load_nir(row["nir_path"])
            channel_arrays.append(nir)
            pad_values.append(114.0 / 255.0)

        if "ndwi" in self.channels:
            ndwi = self.load_ndwi(row["ndwi_path"])
            channel_arrays.append(ndwi)
            pad_values.append(0.0)

        spatial_shapes = [array.shape[:2] for array in channel_arrays]
        if len(set(spatial_shapes)) != 1:
            raise ValueError(
                f"Spatial dimensions do not match for {image_id}: "
                f"{spatial_shapes}"
            )

        image = np.concatenate(channel_arrays, axis=2).astype(np.float32)
        original_height, original_width = image.shape[:2]

        labels = self.load_labels(row["label_path"])

        image, scale, dw, dh = letterbox_multispectral(
            image,
            target_size=self.image_size,
            pad_values=pad_values,
        )

        labels = transform_labels_for_letterbox(
            labels,
            original_width,
            original_height,
            scale,
            dw,
            dh,
            self.image_size,
        )

        if self.augment:
            params = get_augmentation_params(
                image_id=image_id,
                epoch=int(self.epoch.item()),
                horizontal_flip=self.horizontal_flip,
                vertical_flip=self.vertical_flip,
                rotate90=self.rotate90,
                rotation_probability=self.rotation_probability,
            )
            image, labels = apply_augmentation(image, labels, params)

        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        image = torch.from_numpy(image).float()
        labels = torch.from_numpy(labels.astype(np.float32))

        return {
            "images": image,
            "labels": labels,
            "image_id": image_id,

            # Metadata required by the Ultralytics validator.
            # ratio_pad follows the Ultralytics convention:
            #   ((gain_w, gain_h), (pad_w, pad_h))
            "ori_shape": (original_height, original_width),
            "ratio_pad": ((scale, scale), (dw, dh)),
            "im_file": str(row["rgb_path"]),
        }


# ============================================================
# 6. COLLATE FUNCTION
# ============================================================

def multispectral_collate_fn(batch):
    """Stack images and prepend each label row with its batch index."""

    images = torch.stack([item["images"] for item in batch])
    all_labels = []

    for batch_index, item in enumerate(batch):
        labels = item["labels"]

        if len(labels) > 0:
            batch_indices = torch.full(
                (len(labels), 1),
                batch_index,
                dtype=torch.float32,
            )
            labels_with_batch_index = torch.cat(
                [batch_indices, labels],
                dim=1,
            )
            all_labels.append(labels_with_batch_index)

    if all_labels:
        labels = torch.cat(all_labels, dim=0)
    else:
        labels = torch.zeros((0, 6), dtype=torch.float32)

    image_ids = [item["image_id"] for item in batch]

    return {
        "images": images,
        "labels": labels,
        "image_ids": image_ids,
    }