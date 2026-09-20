# SLOT-Align: Distribution Alignment for One-Shot Federated Learning via Optimal Transport

[![Conference](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/virtual/2026/poster/64658)
[![Paper](https://img.shields.io/badge/Paper-OpenReview-b31b1b.svg)](https://openreview.net/forum?id=LChZtIOER6)
[![Code](https://img.shields.io/badge/Code-Available-brightgreen.svg)](https://github.com/daniebera/SLOT-Align)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official repository for the ICML 2026 regular paper:

> **Distribution Alignment for One-Shot Federated Learning via Optimal Transport**  
> Daniele Berardini, Vito Paolo Pastore, Vittorio Murino  
> *Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026.*

[[Paper]](https://openreview.net/forum?id=LChZtIOER6)

## Overview

**SLOT-Align** (*Single-round, Learning-free Optimal Transport Alignment*) is a geometry-aware preprocessing framework for **One-Shot Federated Learning (OSFL)** under domain shift with additional label shift across clients.

Using representations extracted by a shared frozen encoder, SLOT-Align aligns client feature distributions before downstream OSFL training. Clients communicate compact feature statistics, the server constructs a global reference through a **Bures–Wasserstein barycenter**, and local representations are partially transported toward this reference along a Wasserstein geodesic.

SLOT-Align is learning-free and operates directly in feature space, making it suitable as a lightweight alignment stage for OSFL pipelines based on frozen pretrained representations.

## Method

SLOT-Align follows three main steps:

1. **Statistics extraction.** Each client computes first- and second-order statistics of its local feature representations.
2. **Reference construction.** The server computes a global Bures–Wasserstein barycenter from the communicated client statistics.
3. **Feature alignment.** Each client applies an interpolated closed-form Gaussian optimal transport map toward the common reference before downstream one-shot training.

For a client transport map

$$T_k(z) = A_k z + b_k$$

SLOT-Align applies the partial transport

$$T_k^{(\tau)}(z) = (1-\tau)z + \tau T_k(z),$$

where $\tau \in [0,1]$ controls the alignment strength.

Please refer to the paper for the complete formulation and theoretical discussion.

## Supported Scope

This repository provides the clean reference implementation for the experimental setting considered in the paper.

### Datasets

- **Office-Home**
- **DomainNet**
- **Digits**

### Backbones

- **CLIP ViT-B/32** — main backbone
- **ImageNet ViT-B/32** — backbone ablation
- **ImageNet ResNet-18** — backbone ablation

### Downstream OSFL

The repository includes **O-FedAvg** as the downstream one-shot federated-learning baseline.

The additional OSFL methods evaluated in the paper, such as FedCGS and FedPFT, are external methods and remain available through their respective implementations. SLOT-Align can be incorporated as a feature-alignment stage before downstream OSFL training.

## Installation

Python **3.10 or newer** is required.

Clone the repository:

```bash
git clone https://github.com/daniebera/SLOT-Align.git
cd SLOT-Align
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Repository Structure

The pipeline is organized around four main entry points:

```text
prepare_data.py        # dataset preparation and client partitioning
extract_features.py    # frozen-backbone feature extraction
compute_alignment.py   # client statistics, BW barycenter, and OT maps
run_ofedavg.py         # downstream O-FedAvg training and evaluation

slotalign/             # core SLOT-Align implementation
```

Generated data, extracted features, alignment artifacts, models, and results are stored under ignored directories and are not tracked by Git.

## Running SLOT-Align

Run all commands from the repository root.

The following example runs SLOT-Align on **Office-Home** with Dirichlet label heterogeneity \(\alpha=0.1\), CLIP ViT-B/32 features, and final alignment strength \(\tau_{\max}=0.4\).

### 1. Prepare the federated data split

```bash
python prepare_data.py \
    --dataset Office-Home \
    --alpha 0.1 \
    --seed 0
```

### 2. Extract frozen features

```bash
python extract_features.py \
    --dataset Office-Home \
    --alpha 0.1 \
    --backbone ViT-B-32
```

### 3. Compute SLOT-Align

```bash
python compute_alignment.py \
    --dataset Office-Home \
    --alpha 0.1 \
    --backbone ViT-B-32
```

This stage computes the client feature statistics, the global Bures–Wasserstein barycenter, and the client-specific Gaussian optimal transport maps.

### 4. Run downstream O-FedAvg with SLOT-Align

```bash
python run_ofedavg.py \
    --dataset Office-Home \
    --alpha 0.1 \
    --backbone ViT-B-32 \
    --transport-mode scheduled \
    --tau-schedule cosine_anneal \
    --tau-min 0.0 \
    --tau-max 0.4 \
    --communication-rounds 1
```

During one-shot local training, the transport strength is progressively increased from `tau-min = 0.0` to the desired final alignment strength `tau-max`.

For the main SLOT-Align configuration, use:

```text
tau-min = 0.0
tau-max = 0.4
```

Other values of `tau-max` can be used to vary the alignment-strength sensitivity.

With multiple communication rounds, the transport strength is updated once per communication round and remains fixed during the local epochs of that round.

Use

```bash
python <entry-point> --help
```

to inspect all supported options for each stage.

## O-FedAvg without SLOT-Align

The unaligned O-FedAvg baseline can be run directly on the original frozen features:

```bash
python run_ofedavg.py \
    --dataset Office-Home \
    --alpha 0.1 \
    --backbone ViT-B-32 \
    --transport-mode none
```

This configuration requires neither SLOT-Align statistics nor Gaussian transport maps.

## Reproducibility

The pipeline exposes a common random seed controlling the relevant sources of randomness, including dataset partitioning and downstream training.

For reproducible experiments, use the same `--seed` consistently across the relevant stages.

Experiment configurations and evaluation summaries are saved together with the generated results.

All pipeline commands display progress by default. Progress bars can be disabled with:

```bash
--no-progress
```

## Outputs

Training reports per-run, per-round, client/epoch progress, and per-domain evaluation accuracy.

Aggregated metrics are written to `summary.json`.

The quantities used for reporting the paper results are:

- **`macro_accuracy_mean`**: the macro-average accuracy across domains, averaged across runs;
- **`domain_accuracy_std_mean_across_runs`**: the arithmetic mean, across runs, of the within-run standard deviation across domain accuracies.

These provide, respectively, the overall cross-domain performance and its variability across domains.

## Results

SLOT-Align is evaluated on **Office-Home**, **Digits**, and **DomainNet**, showing consistent improvements when integrated with existing OSFL methods under heterogeneous client distributions.

Please refer to the paper for the complete experimental protocol, quantitative comparisons, ablation studies, and additional analyses.

## Citation

If you find this work useful, please cite:

> Daniele Berardini, Vito Paolo Pastore, and Vittorio Murino.  
> **Distribution Alignment for One-Shot Federated Learning via Optimal Transport.**  
> Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026.

BibTeX will be added soon.

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Contact

For questions regarding the paper or implementation, please contact [Daniele Berardini](mailto:daniele.berardini@iit.it).
