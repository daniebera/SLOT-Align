"""Prepare deterministic client partitions for the three paper datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from slotalign.artifacts import ArtifactPaths, save_json
from slotalign.config import DATASETS, dataset_spec, seed_everything, validate_alpha
from slotalign.datasets import (
    dataset_labels,
    digits_dataset,
    domainnet_labels_file,
    office_home_dataset,
)
from slotalign.partitioning import (
    label_skew_fractions,
    retain_by_class_fraction,
    split_train_test_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--dataset", choices=DATASETS, default="DomainNet")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--office-train-fraction", type=float, default=0.7)
    parser.add_argument("--no-progress", action="store_true", help="Hide progress bars.")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not let torchvision download Digits datasets.",
    )
    return parser.parse_args()


def _read_domainnet_records(path: Path) -> tuple[list[str], np.ndarray]:
    records: list[str] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                _, label = line.rsplit(maxsplit=1)
            except ValueError as exc:
                raise ValueError(f"Malformed record at {path}:{line_number}") from exc
            records.append(line)
            labels.append(int(label))
    return records, np.asarray(labels, dtype=np.int64)


def main() -> None:
    args = parse_args()
    alpha = validate_alpha(args.alpha)
    seed_everything(args.seed)
    spec = dataset_spec(args.dataset)
    paths = ArtifactPaths(args.artifact_dir, spec.name, alpha)
    paths.partitions.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    fractions = label_skew_fractions(len(spec.domains), spec.num_classes, alpha, rng)
    client_counts: dict[str, int] = {}

    for client_id, domain in enumerate(
        tqdm(
            spec.domains,
            desc="Preparing partitions",
            unit="domain",
            disable=args.no_progress,
        )
    ):
        domain_dir = paths.partitions / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        if spec.name == "Office-Home":
            dataset = office_home_dataset(args.data_dir, domain)
            labels = dataset_labels(dataset)
            train_indices, test_indices = split_train_test_indices(
                len(dataset), args.office_train_fraction, rng
            )
            client_indices = retain_by_class_fraction(
                train_indices, labels, fractions[client_id], spec.num_classes, rng
            )
            np.save(domain_dir / "client_0_train_indices.npy", client_indices)
            np.save(domain_dir / "test_indices.npy", test_indices)

        elif spec.name == "DomainNet":
            source = domainnet_labels_file(args.data_dir, domain, "train")
            records, labels = _read_domainnet_records(source)
            client_indices = retain_by_class_fraction(
                np.arange(labels.size), labels, fractions[client_id], spec.num_classes, rng
            )
            output = domain_dir / "client_0_train.txt"
            with output.open("w", encoding="utf-8") as stream:
                for index in client_indices:
                    stream.write(records[int(index)] + "\n")

        else:  # Digits
            dataset = digits_dataset(
                args.data_dir,
                domain,
                train=True,
                transform=None,
                download=not args.no_download,
            )
            labels = dataset_labels(dataset)
            client_indices = retain_by_class_fraction(
                np.arange(labels.size), labels, fractions[client_id], spec.num_classes, rng
            )
            np.save(domain_dir / "client_0_train_indices.npy", client_indices)

        if client_indices.size == 0:
            raise ValueError(
                f"Partitioning produced an empty client for {domain}; choose a less extreme alpha or seed"
            )
        client_counts[domain] = int(client_indices.size)

    save_json(
        paths.partitions / "metadata.json",
        {
            "dataset": spec.name,
            "domains": list(spec.domains),
            "num_classes": spec.num_classes,
            "alpha": alpha,
            "seed": args.seed,
            "office_train_fraction": args.office_train_fraction,
            "client_counts": client_counts,
            "class_retention_fractions": fractions.tolist(),
            "partition_semantics": (
                "Each domain is one client. For alpha>0, each class receives a Dirichlet "
                "draw across domain clients and each domain retains that fraction of its local class."
            ),
        },
    )
    print(f"Saved deterministic partitions to {paths.partitions}")


if __name__ == "__main__":
    main()
