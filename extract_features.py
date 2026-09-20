"""Extract frozen train/test features for prepared paper datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from slotalign.artifacts import ArtifactPaths, save_json
from slotalign.config import BACKBONES, DATASETS, dataset_spec, seed_everything, validate_alpha
from slotalign.datasets import (
    DomainNetDataset,
    digits_dataset,
    domainnet_labels_file,
    office_home_dataset,
    subset,
)
from slotalign.features import create_backbone, extract_features, save_feature_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--dataset", choices=DATASETS, default="DomainNet")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--backbone", choices=BACKBONES, default="ViT-B-32")
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Hide progress bars.")
    return parser.parse_args()


def _load_split_dataset(args, paths, domain, split, preprocess):
    if args.dataset == "Office-Home":
        dataset = office_home_dataset(args.data_dir, domain, transform=preprocess)
        filename = "client_0_train_indices.npy" if split == "train" else "test_indices.npy"
        index_path = paths.partitions / domain / filename
        if not index_path.is_file():
            raise FileNotFoundError(f"Run prepare_data.py first; missing {index_path}")
        return subset(dataset, np.load(index_path))

    if args.dataset == "DomainNet":
        labels_path = (
            paths.partitions / domain / "client_0_train.txt"
            if split == "train"
            else domainnet_labels_file(args.data_dir, domain, "test")
        )
        return DomainNetDataset(args.data_dir, labels_path, transform=preprocess)

    dataset = digits_dataset(
        args.data_dir,
        domain,
        train=split == "train",
        transform=preprocess,
        download=not args.no_download,
    )
    if split == "test":
        return dataset
    index_path = paths.partitions / domain / "client_0_train_indices.npy"
    if not index_path.is_file():
        raise FileNotFoundError(f"Run prepare_data.py first; missing {index_path}")
    return subset(dataset, np.load(index_path))


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative")
    alpha = validate_alpha(args.alpha)
    seed_everything(args.seed)
    spec = dataset_spec(args.dataset)
    paths = ArtifactPaths(args.artifact_dir, spec.name, alpha, args.backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.backbone} on {device}...", flush=True)
    model, preprocess, encode = create_backbone(args.backbone)
    splits = ("train", "test") if args.split == "both" else (args.split,)

    feature_dim = None
    counts: dict[str, dict[str, int]] = {}
    for domain in spec.domains:
        counts[domain] = {}
        for split in splits:
            print(f"Loading {domain} {split} images...", flush=True)
            dataset = _load_split_dataset(args, paths, domain, split, preprocess)
            features, labels = extract_features(
                dataset,
                model,
                encode,
                device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                progress=not args.no_progress,
                progress_desc=f"{domain} {split} features",
            )
            feature_dim = int(features.shape[1])
            output_root = paths.train_features if split == "train" else paths.test_features
            filename = "client_0.npz" if split == "train" else "test.npz"
            print(f"Saving {domain} {split} features...", flush=True)
            save_feature_set(output_root / domain / filename, features, labels)
            counts[domain][split] = int(labels.size)
            print(f"{domain} {split}: {features.shape}")

    save_json(
        paths.backbone_root / "features" / "metadata.json",
        {
            "dataset": spec.name,
            "alpha": alpha,
            "backbone": args.backbone,
            "feature_dim": feature_dim,
            "seed": args.seed,
            "splits": list(splits),
            "counts": counts,
        },
    )


if __name__ == "__main__":
    main()
