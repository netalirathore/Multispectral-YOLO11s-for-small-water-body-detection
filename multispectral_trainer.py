import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ultralytics.models.yolo.detect import (
    DetectionTrainer,
    DetectionValidator,
)
from ultralytics.utils import LOGGER
from ultralytics_dataset_adapter import (
    UltralyticsMultispectralDataset,
    MultispectralDataLoader,
)

# ---------------------------------------------------------
# 1. Import project modules
# ---------------------------------------------------------

MODULE_DIR = "/home/netali/projects/Multispectral/py_files"

if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

import multispectral_moudule as mm

from multispectral_model import initialize_multispectral_yolo
from ultralytics_dataset_adapter import UltralyticsMultispectralDataset


def _rgb_only(batch):
    """Shallow copy of a batch whose image tensor keeps only RGB (for plotting)."""
    batch = dict(batch)
    batch["img"] = batch["img"][:, :3]
    return batch


# ---------------------------------------------------------
# 2. Custom multispectral validator
# ---------------------------------------------------------

class MultispectralDetectionValidator(DetectionValidator):

    def __init__(
        self,
        *args,
        experiment_name,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.experiment_name = experiment_name

    def get_dataloader(self, dataset_path, batch_size):

        dataset = mm.MultispectralYOLODataset(
            experiment_name=self.experiment_name,
            split="val",
            image_size=self.args.imgsz,
            augment=False,
        )

        dataset = UltralyticsMultispectralDataset(dataset)

        return MultispectralDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.args.workers > 0,
            collate_fn=UltralyticsMultispectralDataset.collate_fn,
        )

    def preprocess(self, batch):

        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(
                    self.device,
                    non_blocking=self.device.type not in {"cpu", "mps"},
                )

        # Images are already normalized. Do NOT divide by 255.
        # During training-time validation the model may be FP16, so
        # match the input dtype to the model actually being validated.
        dtype = torch.float32
        trainer = getattr(self, "mm_trainer", None)
        if trainer is not None:
            model = trainer.ema.ema if trainer.ema else trainer.model
            dtype = next(model.parameters()).dtype
        batch["img"] = batch["img"].to(dtype)

        return batch

    # Plotting helpers only understand 3-channel images.
    def plot_val_samples(self, batch, *args, **kwargs):
        return super().plot_val_samples(_rgb_only(batch), *args, **kwargs)

    def plot_predictions(self, batch, *args, **kwargs):
        return super().plot_predictions(_rgb_only(batch), *args, **kwargs)


# ---------------------------------------------------------
# 3. Custom multispectral trainer
# ---------------------------------------------------------

class MultispectralDetectionTrainer(DetectionTrainer):

    def __init__(
        self,
        experiment_name,
        overrides=None,
    ):

        if experiment_name not in mm.EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment: {experiment_name}. "
                f"Available: {mm.EXPERIMENTS}"
            )

        self.experiment_name = experiment_name

        self.input_channels = sum(
            3 if channel == "rgb" else 1
            for channel in mm.EXPERIMENT_CHANNELS[experiment_name]
        )

        overrides = dict(overrides or {})

        # Keep the model configuration explicit.
        overrides.setdefault("model", "yolo11s.pt")
        overrides.setdefault("data", "multispectral")
        overrides.setdefault("task", "detect")

        super().__init__(overrides=overrides)

    # -----------------------------------------------------
    # Dataset metadata
    # -----------------------------------------------------

    def get_dataset(self):

        names = dict(mm.CLASS_NAMES)

        return {
            "train": "multispectral_train",
            "val": "multispectral_val",
            "nc": len(names),
            "names": names,
            "channels": self.input_channels,
        }

    # -----------------------------------------------------
    # Initialize pretrained multispectral YOLO11s
    # -----------------------------------------------------

    def setup_model(self):

        self.model = initialize_multispectral_yolo(
            weights=self.args.model,
            input_channels=self.input_channels,
            num_classes=self.data["nc"],
            device=None,
        )

        # Use the trainer's configuration for the native loss.
        self.model.args = self.args
        self.model.names = dict(self.data["names"])
        self.model.task = "detect"

        print("\nMultispectral model initialized")
        print("Experiment:", self.experiment_name)
        print("Input channels:", self.input_channels)
        print("Number of classes:", self.data["nc"])

    # -----------------------------------------------------
    # Build custom training / validation dataloaders
    # -----------------------------------------------------

    def get_dataloader(
        self,
        dataset_path,
        batch_size=16,
        rank=0,
        mode="train",
    ):

        if mode not in {"train", "val"}:
            raise ValueError(f"Unsupported mode: {mode}")

        split = "train" if mode == "train" else "val"

        dataset = mm.MultispectralYOLODataset(
            experiment_name=self.experiment_name,
            split=split,
            image_size=self.args.imgsz,
            augment=(mode == "train"),
        )

        dataset = UltralyticsMultispectralDataset(dataset)

        return MultispectralDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(mode == "train"),
            num_workers=self.args.workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.args.workers > 0,
            drop_last=False,
            collate_fn=UltralyticsMultispectralDataset.collate_fn,
        )

    # -----------------------------------------------------
    # Preprocess training batches
    # -----------------------------------------------------

    def preprocess_batch(self, batch):

        for key, value in batch.items():

            if isinstance(value, torch.Tensor):

                batch[key] = value.to(
                    self.device,
                    non_blocking=self.device.type not in {"cpu", "mps"},
                )

        # RGB and NIR are already scaled to [0, 1].
        # NDWI is retained in its original range.
        # Never divide the complete tensor by 255.

        batch["img"] = batch["img"].float()

        return batch

    # -----------------------------------------------------
    # Update deterministic augmentation epoch
    # -----------------------------------------------------

    def run_callbacks(self, event):

        super().run_callbacks(event)

        if event == "on_train_epoch_start":

            if hasattr(self, "train_loader"):

                adapter = self.train_loader.dataset
                dataset = adapter.dataset

                dataset.set_epoch(self.epoch)

    # -----------------------------------------------------
    # Plotting (RGB channels only)
    # -----------------------------------------------------

    def plot_training_samples(self, batch, *args, **kwargs):
        return super().plot_training_samples(
            _rgb_only(batch), *args, **kwargs
        )

    # -----------------------------------------------------
    # Final evaluation
    # -----------------------------------------------------

    def final_eval(self):
        """
        The stock final_eval re-validates best.pt through a standalone
        validator, which calls check_det_dataset("multispectral") and
        fails because there is no dataset YAML. Strip optimizer state
        and skip that step.
        """
        from ultralytics.utils.torch_utils import strip_optimizer

        for f in (self.last, self.best):
            if f.exists():
                strip_optimizer(f)

        LOGGER.info(
            "Standalone final validation skipped "
            "(custom multispectral dataset)."
        )

    # -----------------------------------------------------
    # Use our custom validator
    # -----------------------------------------------------

    def get_validator(self):

        # Names must match the order of the loss vector.
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"

        validator = MultispectralDetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=self.args,
            _callbacks=self.callbacks,
            experiment_name=self.experiment_name,
        )
        validator.mm_trainer = self
        return validator