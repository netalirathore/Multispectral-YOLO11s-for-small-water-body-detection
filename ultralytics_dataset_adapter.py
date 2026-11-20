import torch
from torch.utils.data import Dataset

class MultispectralDataLoader(torch.utils.data.DataLoader):

    def reset(self):
        """Compatibility method required by Ultralytics."""
        pass


class UltralyticsMultispectralDataset(Dataset):
    """
    Adapter between MultispectralYOLODataset and
    the Ultralytics YOLO detection training pipeline.
    """

    def __init__(self, multispectral_dataset):
        self.dataset = multispectral_dataset

    def __len__(self):
        return len(self.dataset)

    @property
    def labels(self):
        """
        Label metadata for Ultralytics plot_training_labels().
        Reads only the label files (no image loading).
        """
        all_labels = []

        for label_path in self.dataset.df["label_path"]:
            labels = self.dataset.load_labels(label_path)
            all_labels.append({
                "cls": labels[:, 0:1],
                "bboxes": labels[:, 1:5],
            })

        return all_labels

    def __getitem__(self, index):
        sample = self.dataset[index]

        image = sample["images"]
        labels = sample["labels"]

        return {
            "img": image,
            "cls": labels[:, 0:1].float(),
            "bboxes": labels[:, 1:5].float(),
            "image_id": sample["image_id"],

            # Validation metadata
            "ori_shape": sample["ori_shape"],
            "ratio_pad": sample["ratio_pad"],
            "im_file": sample["im_file"],
        }

    @staticmethod
    def collate_fn(batch):
        """
        Combine samples into an Ultralytics-compatible batch.
        """

        images = torch.stack(
            [sample["img"] for sample in batch],
            dim=0
        )

        classes = []
        boxes = []
        batch_indices = []

        for index, sample in enumerate(batch):
            cls = sample["cls"]
            bbox = sample["bboxes"]

            classes.append(cls)
            boxes.append(bbox)

            batch_indices.append(
                torch.full(
                    (len(cls),),
                    index,
                    dtype=torch.int64
                )
            )

        if classes:
            classes = torch.cat(classes, dim=0)
            boxes = torch.cat(boxes, dim=0)
            batch_indices = torch.cat(batch_indices, dim=0)
        else:
            classes = torch.empty(
                (0, 1), dtype=torch.float32
            )
            boxes = torch.empty(
                (0, 4), dtype=torch.float32
            )
            batch_indices = torch.empty(
                (0,), dtype=torch.int64
            )

        return {
            "img": images,
            "cls": classes,
            "bboxes": boxes,
            "batch_idx": batch_indices,

            # Metadata needed by DetectionValidator
            "ori_shape": [
                sample["ori_shape"] for sample in batch
            ],
            "ratio_pad": [
                sample["ratio_pad"] for sample in batch
            ],
            "im_file": [
                sample["im_file"] for sample in batch
            ],
        }