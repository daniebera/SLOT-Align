"""Dataset adapters for Office-Home, DomainNet, and Digits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder, MNIST, SVHN, USPS


def ensure_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


class DomainNetDataset(Dataset):
    def __init__(self, data_root: Path, labels_file: Path, transform=None):
        self.image_root = data_root / "DomainNet"
        self.transform = transform
        self.image_paths: list[Path] = []
        self.labels: list[int] = []

        with labels_file.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    relative_path, label = line.rsplit(maxsplit=1)
                except ValueError as exc:
                    raise ValueError(
                        f"Malformed DomainNet record at {labels_file}:{line_number}"
                    ) from exc
                self.image_paths.append(self.image_root / relative_path)
                self.labels.append(int(label))

        missing = [path for path in self.image_paths if not path.is_file()]
        if missing:
            preview = ", ".join(str(path) for path in missing[:3])
            raise FileNotFoundError(f"Missing DomainNet images, including: {preview}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        with self.image_paths[index].open("rb") as stream:
            image = Image.open(stream).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


class SynthDigitsDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            try:
                label = int(class_dir.name)
            except ValueError as exc:
                raise ValueError(f"Synthetic digit class folder must be numeric: {class_dir}") from exc
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file():
                    self.samples.append((image_path, label))
        self.targets = np.asarray([label for _, label in self.samples], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with path.open("rb") as stream:
            image = Image.open(stream).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def office_home_dataset(data_root: Path, domain: str, transform=None) -> ImageFolder:
    return ImageFolder(data_root / "Office-Home" / domain, transform=transform)


def domainnet_labels_file(data_root: Path, domain: str, split: str) -> Path:
    return data_root / "DomainNet" / f"{domain}_{split}.txt"


def digits_dataset(
    data_root: Path,
    domain: str,
    train: bool,
    transform=None,
    download: bool = True,
):
    root = data_root / "Digits" / domain
    if domain == "mnist":
        return MNIST(root, train=train, download=download, transform=transform)
    if domain == "usps":
        return USPS(root, train=train, download=download, transform=transform)
    if domain == "svhn":
        return SVHN(
            root, split="train" if train else "test", download=download, transform=transform
        )
    if domain == "synth":
        return SynthDigitsDataset(root / ("train" if train else "test"), transform=transform)
    raise ValueError(f"Unsupported Digits domain: {domain}")


def dataset_labels(dataset) -> np.ndarray:
    for attribute in ("targets", "labels"):
        if hasattr(dataset, attribute):
            labels = getattr(dataset, attribute)
            if hasattr(labels, "detach"):
                labels = labels.detach().cpu().numpy()
            return np.asarray(labels, dtype=np.int64)
    if hasattr(dataset, "samples"):
        return np.asarray([sample[1] for sample in dataset.samples], dtype=np.int64)
    raise TypeError(f"Cannot obtain labels from {type(dataset).__name__}")


def subset(dataset, indices: np.ndarray) -> Subset:
    return Subset(dataset, np.asarray(indices, dtype=np.int64).tolist())
