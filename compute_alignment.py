"""Compute SLOT-Align client statistics, barycenter, and Gaussian maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from slotalign.alignment import compute_gaussian_map, wasserstein_barycenter
from slotalign.artifacts import ArtifactPaths, load_json, save_json
from slotalign.config import BACKBONES, DATASETS, dataset_spec, validate_alpha
from slotalign.features import load_feature_set
from slotalign.statistics import GlobalPCA, fit_global_pca, fit_ledoit_wolf_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--dataset", choices=DATASETS, default="DomainNet")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--backbone", choices=BACKBONES, default="ViT-B-32")
    parser.add_argument(
        "--statistics",
        choices=("lw", "pca_lw"),
        default="lw",
        help="Paper-default full-dimensional LW, or optional common-PCA then LW.",
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.99,
        help="Explained variance retained by optional common PCA.",
    )
    parser.add_argument("--spd-epsilon", type=float, default=1e-8)
    parser.add_argument("--no-progress", action="store_true", help="Hide progress bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alpha = validate_alpha(args.alpha)
    if args.statistics == "pca_lw" and not 0.0 < args.pca_variance <= 1.0:
        raise ValueError("pca-variance must lie in (0, 1]")
    spec = dataset_spec(args.dataset)
    paths = ArtifactPaths(args.artifact_dir, spec.name, alpha, args.backbone)
    alignment_dir = paths.alignment_dir(args.statistics, args.pca_variance)
    partition_metadata = load_json(paths.partitions / "metadata.json")
    feature_metadata = load_json(paths.backbone_root / "features" / "metadata.json")
    if partition_metadata.get("dataset") != spec.name or partition_metadata.get("alpha") != alpha:
        raise ValueError("Partition metadata does not match the requested dataset/alpha")
    if (
        feature_metadata.get("dataset") != spec.name
        or feature_metadata.get("alpha") != alpha
        or feature_metadata.get("backbone") != args.backbone
    ):
        raise ValueError("Feature metadata does not match the requested configuration")
    original_features = [
        load_feature_set(paths.train_features / domain / "client_0.npz")[0]
        for domain in tqdm(
            spec.domains,
            desc="Loading client features",
            unit="client",
            disable=args.no_progress,
        )
    ]
    original_dim = int(original_features[0].shape[1])
    if any(features.shape[1] != original_dim for features in original_features):
        raise ValueError("Feature artifacts have inconsistent encoder dimensions")

    pca: GlobalPCA | None = None
    client_features = original_features
    if args.statistics == "pca_lw":
        print("Fitting global PCA...", flush=True)
        pca = fit_global_pca(original_features, explained_variance=args.pca_variance)
        pca.save(alignment_dir / "global_pca.npz")
        client_features = [
            pca.transform(features)
            for features in tqdm(
                original_features,
                desc="Projecting clients",
                unit="client",
                disable=args.no_progress,
            )
        ]

    client_statistics = [
        fit_ledoit_wolf_statistics(features, spd_epsilon=args.spd_epsilon)
        for features in tqdm(
            client_features,
            desc="Estimating client statistics",
            unit="client",
            disable=args.no_progress,
        )
    ]
    print("Computing Bures-Wasserstein barycenter...", flush=True)
    barycenter_mean, barycenter_covariance, weights = wasserstein_barycenter(client_statistics)
    alignment_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        alignment_dir / "barycenter.npz",
        mean=barycenter_mean,
        covariance=barycenter_covariance,
        weights=weights,
    )

    clients = []
    for client_id, (domain, statistics) in enumerate(
        tqdm(
            zip(spec.domains, client_statistics, strict=True),
            total=len(spec.domains),
            desc="Computing transport maps",
            unit="client",
            disable=args.no_progress,
        )
    ):
        domain_dir = alignment_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            domain_dir / "client_0_statistics.npz",
            count=np.asarray(statistics.count),
            mean=statistics.mean,
            covariance=statistics.covariance,
        )
        transport = compute_gaussian_map(
            statistics.mean,
            statistics.covariance,
            barycenter_mean,
            barycenter_covariance,
        )
        mapped_mean = statistics.mean @ transport.matrix.T + transport.bias
        mapped_covariance = transport.matrix @ statistics.covariance @ transport.matrix.T
        if not np.allclose(mapped_mean, barycenter_mean, rtol=1e-5, atol=1e-7):
            raise RuntimeError(f"Mean transport verification failed for {domain}")
        if not np.allclose(mapped_covariance, barycenter_covariance, rtol=2e-4, atol=1e-6):
            raise RuntimeError(f"Covariance transport verification failed for {domain}")
        transport.save(domain_dir / "client_0_map.npz")
        clients.append(
            {"client_id": client_id, "domain": domain, "sample_count": statistics.count}
        )

    save_json(
        alignment_dir / "metadata.json",
        {
            "dataset": spec.name,
            "alpha": alpha,
            "backbone": args.backbone,
            "statistics": args.statistics,
            "pca_variance": args.pca_variance if pca is not None else None,
            "encoder_feature_dim": original_dim,
            "alignment_feature_dim": int(client_features[0].shape[1]),
            "spd_epsilon": args.spd_epsilon,
            "partition_seed": partition_metadata.get("seed"),
            "feature_seed": feature_metadata.get("seed"),
            "barycenter_weighting": "sample_count",
            "map_orientation": "client_source_to_global_barycenter_target",
            "clients": clients,
        },
    )
    print(f"Saved SLOT-Align artifacts to {alignment_dir}")


if __name__ == "__main__":
    main()
