"""Run O-FedAvg with disabled, fixed, or scheduled SLOT-Align transport."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from slotalign.alignment import GaussianMap, apply_partial_transport
from slotalign.artifacts import ArtifactPaths, load_json, save_json
from slotalign.config import (
    BACKBONES,
    DATASETS,
    dataset_spec,
    seed_everything,
    validate_alpha,
    validate_tau,
)
from slotalign.features import load_feature_set
from slotalign.federated import (
    ClientTrainingData,
    evaluate_classifier,
    train_federated_classifier,
)
from slotalign.statistics import GlobalPCA
from slotalign.schedules import TAU_SCHEDULES, compute_tau_schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--dataset", choices=DATASETS, default="DomainNet")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--backbone", choices=BACKBONES, default="ViT-B-32")
    parser.add_argument("--statistics", choices=("lw", "pca_lw"), default="lw")
    parser.add_argument("--pca-variance", type=float, default=0.99)
    parser.add_argument(
        "--transport-mode",
        choices=("none", "fixed", "scheduled"),
        default="fixed",
        help=(
            "Disable SLOT-Align transport, use a fixed tau, or use the historical "
            "adaptive tau schedule."
        ),
    )
    parser.add_argument("--tau", type=float, default=0.4)
    parser.add_argument(
        "--tau-schedule",
        "--tau_schedule",
        dest="tau_schedule",
        choices=TAU_SCHEDULES,
        default="cosine_anneal",
        help="Schedule used only with --transport-mode scheduled.",
    )
    parser.add_argument(
        "--tau-max",
        "--tau_max",
        dest="tau_max",
        type=float,
        default=0.0,
        help="Upper schedule bound; historical default is 0.0.",
    )
    parser.add_argument(
        "--tau-min",
        "--tau_min",
        dest="tau_min",
        type=float,
        default=0.0,
        help="Lower schedule bound; historical default is 0.0.",
    )
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--local-epochs", type=int, default=10)
    parser.add_argument("--communication-rounds", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-progress", action="store_true", help="Hide progress bars.")
    return parser.parse_args()


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def summarize_runs(run_results: list[dict], domains: tuple[str, ...]) -> dict:
    """Summarize final-model accuracies without changing per-run metrics."""

    macro_scores = np.asarray([result["macro_accuracy"] for result in run_results])
    domain_std_scores = np.asarray(
        [result["domain_accuracy_std"] for result in run_results], dtype=np.float64
    )
    domain_summary = {}
    for domain in domains:
        scores = np.asarray(
            [result["domain_accuracy"][domain] for result in run_results],
            dtype=np.float64,
        )
        domain_summary[domain] = {
            "mean": float(scores.mean()),
            "std_across_runs": float(scores.std()),
        }
    mean_domain_scores = np.asarray(
        [domain_summary[domain]["mean"] for domain in domains], dtype=np.float64
    )
    return {
        "macro_accuracy_mean": float(macro_scores.mean()),
        "macro_accuracy_std_across_runs": float(macro_scores.std()),
        "domain_accuracy_std_mean_across_runs": float(domain_std_scores.mean()),
        "std_across_domain_means": float(mean_domain_scores.std()),
        "domain_accuracy": domain_summary,
        "runs": run_results,
    }


def main() -> None:
    args = parse_args()
    alpha = validate_alpha(args.alpha)
    if min(args.batch_size, args.local_epochs, args.communication_rounds, args.runs) <= 0:
        raise ValueError(
            "batch-size, local-epochs, communication-rounds, and runs must be positive"
        )
    if args.transport_mode == "fixed":
        training_tau = validate_tau(args.tau)
        tau_values = None
        evaluation_tau = training_tau
    elif args.transport_mode == "scheduled":
        tau_max = validate_tau(args.tau_max)
        tau_min = validate_tau(args.tau_min)
        schedule_steps = (
            args.local_epochs
            if args.communication_rounds == 1
            else args.communication_rounds
        )
        tau_values = compute_tau_schedule(
            args.tau_schedule, schedule_steps, tau_max, tau_min
        )
        training_tau = tau_values[0]
        evaluation_tau = tau_max
    else:
        training_tau = 0.0
        tau_values = None
        evaluation_tau = 0.0
    if (
        args.transport_mode != "none"
        and args.statistics == "pca_lw"
        and not 0.0 < args.pca_variance <= 1.0
    ):
        raise ValueError("pca-variance must lie in (0, 1]")
    spec = dataset_spec(args.dataset)
    paths = ArtifactPaths(args.artifact_dir, spec.name, alpha, args.backbone)
    alignment_dir = None
    metadata = None
    pca = None
    if args.transport_mode != "none":
        alignment_dir = paths.alignment_dir(args.statistics, args.pca_variance)
        metadata = load_json(alignment_dir / "metadata.json")
        for key, expected in (
            ("dataset", spec.name),
            ("alpha", alpha),
            ("backbone", args.backbone),
        ):
            if metadata.get(key) != expected:
                raise ValueError(
                    f"Alignment metadata mismatch for {key}: "
                    f"{metadata.get(key)!r} != {expected!r}"
                )
        if metadata["statistics"] != args.statistics:
            raise ValueError("Requested statistics mode does not match alignment artifacts")
        if metadata["statistics"] == "pca_lw":
            if metadata.get("pca_variance") != args.pca_variance:
                raise ValueError("Requested PCA variance does not match alignment artifacts")
            pca = GlobalPCA.load(alignment_dir / "global_pca.npz")

    clients: list[ClientTrainingData] = []
    test_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for domain in spec.domains:
        suffix = " and alignment" if alignment_dir is not None else ""
        print(f"Loading {domain} features{suffix}...", flush=True)
        train_features, train_labels = load_feature_set(
            paths.train_features / domain / "client_0.npz"
        )
        test_features, test_labels = load_feature_set(paths.test_features / domain / "test.npz")
        if pca is not None:
            train_features = pca.transform(train_features)
            test_features = pca.transform(test_features)
        transport = (
            GaussianMap.load(alignment_dir / domain / "client_0_map.npz")
            if alignment_dir is not None
            else None
        )
        if transport is not None and train_features.shape[1] != transport.dimension:
            raise ValueError(f"Feature/map dimension mismatch for {domain}")
        clients.append(
            ClientTrainingData(domain, train_features, train_labels, transport)
        )
        if transport is not None:
            test_features = apply_partial_transport(
                test_features, transport, evaluation_tau
            )
        test_sets[domain] = (test_features.astype(np.float32, copy=False), test_labels)

    device = _select_device(args.device)
    run_results = []
    if args.transport_mode == "none":
        transport_path = Path("no_transport")
    elif args.transport_mode == "fixed":
        transport_path = Path(alignment_dir.name) / f"tau_{training_tau:g}"
    else:
        transport_path = Path(alignment_dir.name) / (
            f"{args.tau_schedule}_tmax_{evaluation_tau:g}_tmin_{args.tau_min:g}"
        )
        print(
            f"Tau schedule ({'local epochs' if args.communication_rounds == 1 else 'rounds'}): "
            f"{tau_values}"
        )
    output_root = (
        args.output_dir
        / spec.name
        / format(alpha, ".12g")
        / args.backbone
        / transport_path
        / (
            f"{args.optimizer}_lr_{args.learning_rate:g}_batch_{args.batch_size}_"
            f"local_{args.local_epochs}_rounds_{args.communication_rounds}_seed_{args.seed}"
        )
    )
    for run_index in range(args.runs):
        run_seed = args.seed + run_index
        print(f"Run {run_index + 1}/{args.runs} (seed {run_seed})", flush=True)
        seed_everything(run_seed)
        classifier, history = train_federated_classifier(
            clients=clients,
            num_classes=spec.num_classes,
            communication_rounds=args.communication_rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            optimizer_name=args.optimizer,
            learning_rate=args.learning_rate,
            tau=training_tau,
            seed=run_seed,
            device=device,
            tau_schedule=tau_values,
            progress=not args.no_progress,
            progress_prefix=f"Run {run_index + 1}/{args.runs}",
        )
        domain_accuracy = {}
        for domain, (features, labels) in test_sets.items():
            domain_accuracy[domain] = evaluate_classifier(
                classifier,
                features,
                labels,
                args.batch_size,
                device,
                progress=not args.no_progress,
                progress_desc=f"Run {run_index + 1}/{args.runs} evaluating {domain}",
            )
            print(f"  {domain}: {domain_accuracy[domain]:.4f}", flush=True)
        values = np.asarray(list(domain_accuracy.values()), dtype=np.float64)
        result = {
            "run": run_index,
            "seed": run_seed,
            "domain_accuracy": domain_accuracy,
            "macro_accuracy": float(values.mean()),
            "domain_accuracy_std": float(values.std()),
            "training_history": history,
        }
        run_results.append(result)
        run_dir = output_root / f"run_{run_index}"
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {key: value.detach().cpu().clone() for key, value in classifier.state_dict().items()},
            run_dir / "final_classifier.pt",
        )
        save_json(run_dir / "metrics.json", result)

    effective_config = {
        "dataset": spec.name,
        "alpha": alpha,
        "backbone": args.backbone,
        "classifier": "linear",
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "local_epochs": args.local_epochs,
        "communication_rounds": args.communication_rounds,
        "transport_mode": args.transport_mode,
        "tau": training_tau if args.transport_mode == "fixed" else None,
        "tau_schedule": args.tau_schedule if args.transport_mode == "scheduled" else None,
        "tau_min": args.tau_min if args.transport_mode == "scheduled" else None,
        "tau_max": evaluation_tau if args.transport_mode == "scheduled" else None,
        "tau_values": tau_values,
        "tau_schedule_scope": (
            "local_epoch"
            if args.transport_mode == "scheduled" and args.communication_rounds == 1
            else "communication_round"
            if args.transport_mode == "scheduled"
            else None
        ),
        "evaluation_tau": evaluation_tau,
        "statistics": metadata["statistics"] if metadata is not None else None,
        "pca_variance": metadata.get("pca_variance") if metadata is not None else None,
        "partition_seed": metadata.get("partition_seed") if metadata is not None else None,
        "feature_seed": metadata.get("feature_seed") if metadata is not None else None,
        "seed": args.seed,
        "runs": args.runs,
        "device": str(device),
        "aggregation_weighting": "equal_client",
        "checkpoint_selection": "final_model_no_test_selection",
    }
    save_json(output_root / "config.json", effective_config)
    save_json(output_root / "summary.json", summarize_runs(run_results, spec.domains))
    print(f"Saved coherent final-model results to {output_root}")


if __name__ == "__main__":
    main()
