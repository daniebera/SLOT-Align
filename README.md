# SLOT-Align: Distribution Alignment for One-Shot Federated Learning via Optimal Transport

[![Conference](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/)
[![Paper](https://img.shields.io/badge/Paper-OpenReview-b31b1b.svg)](https://openreview.net/forum?id=LChZtIOER6)
[![Code](https://img.shields.io/badge/Code-Coming%20Soon-lightgrey.svg)](#code-release)

Official repository for the ICML 2026 regular paper:

> **Distribution Alignment for One-Shot Federated Learning via Optimal Transport**  
> Daniele Berardini, Vito Paolo Pastore, Vittorio Murino  
> *Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026.*

[[Paper]](https://openreview.net/forum?id=LChZtIOER6)

## Overview

**SLOT-Align** (*Single-round, Learning-free Optimal Transport Alignment*) is a geometry-aware preprocessing framework for **One-Shot Federated Learning (OSFL)** under domain shift with additional label shift across clients.

Using representations extracted by a shared frozen encoder, SLOT-Align aligns client feature distributions before downstream OSFL training. Clients communicate compact feature statistics, the server constructs a global reference through a **Bures–Wasserstein barycenter**, and local representations are partially transported toward this reference along a Wasserstein geodesic.

## Method

SLOT-Align follows three steps:

1. **Statistics extraction:** each client computes first- and second-order statistics of its local feature representations.
2. **Reference construction:** the server computes a global Bures–Wasserstein barycenter from the communicated statistics.
3. **Feature alignment:** each client applies an interpolated closed-form optimal transport map before downstream one-shot training.

The framework is learning-free, requires a single exchange of compact statistics, and can be integrated with OSFL pipelines based on frozen pretrained encoders.

## Results

SLOT-Align is evaluated on **Office-Home**, **Digits**, and **DomainNet**, showing consistent improvements when integrated with existing OSFL methods under heterogeneous client distributions. Please refer to the paper for the complete experimental protocol, quantitative results, and ablation studies.

## Code Release

Code and reproduction instructions will be released soon.

## Citation

BibTeX will be added soon.

## License

Licensing terms for the source-code release will be specified when the implementation is made available.

## Contact

For questions regarding the paper or the future code release, please contact [Daniele Berardini](mailto:daniele.berardini@iit.it).
