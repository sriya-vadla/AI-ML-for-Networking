# AI‑ML‑for‑Networking

> **Intelligent traffic classification & anomaly detection for modern, encrypted networks.**
>
> • **Random Forest** flow classifier • **Isolation Forest** zero‑day detector • **Privacy‑aware feature pipeline**

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Project Structure](#project-structure)
6. [Usage](#usage)
7. [Sample Results](#sample-results)
8. [Roadmap](#roadmap)
9. [Contributing](#contributing)
10. [License](#license)

---

## Overview

`AI‑ML‑for‑Networking` is a Python toolkit that tackles the blind‑spot problem created by ubiquitous TLS, QUIC, and bursty IoT traffic.\
It combines classical ensemble learners with lightweight preprocessing to:

- **Classify** network flows as *benign* or *malicious* (multi‑class labels supported).
- **Detect** never‑before‑seen (zero‑day) anomalies in real time.
- **Preserve privacy** by aggregating sensitive packet fields on‑prem while exporting only derived features.

Benchmarked on the NSL‑KDD dataset, the reference pipeline achieves **Macro‑F1 ≈ 0.92** and processes \~10 k flows / sec on a laptop‑class CPU.

---

## Key Features

- 🔍 **Supervised Flow Classification** — Grid‑search‑tuned Random Forest (50‑100 trees, depth ≤ 10).
- 🚨 **Unsupervised Anomaly Detection** — Isolation Forest with contamination ≤ 1 %.
- 🗄️ **Modular Codebase** — swap models or datasets via config flags; fully reproducible.
- ⚡ **Resource Friendly** — runs in ≤ 2 GB RAM; no GPU required.
- 🔒 **Privacy Layer** — optional feature aggregator to keep raw packet headers on‑prem.

---

## Installation

```bash
# 1. Clone the repo
$ git clone https://github.com/sriya-vadla/AI-ML-for-Networking.git
$ cd AI-ML-for-Networking

# 2. Create a virtual env (recommended)
$ python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
$ pip install -r requirements.txt
```

> **Dataset** — NSL‑KDD ARFF files are auto‑downloaded on first run.\
> To use your own PCAP/CSV flows, see [`docs/DATASET.md`](docs/DATASET.md).

---

## Quick Start

Train the classifier and anomaly detector, then run predictions:

```bash
# Train & evaluate (creates models/*.joblib)
$ python run_analysis.py --mode train

# Predict on a CSV of flow features
$ python run_analysis.py --mode predict --input sample/flows.csv --output predictions.csv
```

---

## Project Structure

```text
AI-ML-for-Networking/
│
├── preprocessing.py       # Data cleaning & feature engineering
├── models.py              # ML pipelines & hyper‑parameter search
├── run_analysis.py        # Orchestration CLI
│
├── sample/                # Demo configs & example flows
├── tests/                 # Unit & integration tests (pytest)
├── requirements.txt       # Python dependencies
└── docs/                  # Extended documentation & diagrams
    └── DATASET.md         # How to use custom datasets
```

---

## Usage

Common CLI flags:

```bash
--mode {train,predict,benchmark}
--input  path/to/flows.csv      # required for predict/benchmark
--output path/to/out.csv        # default: predictions.csv
--model_dir models/             # where .joblib files live
--log_level {info,debug}
```

See `python run_analysis.py --help` for the full list.

### Live Capture (experimental)

```bash
$ python run_analysis.py --mode live --iface eth0
```

> Requires `scapy` or `pyshark`. Packets are aggregated into flow features before inference.

---

## Sample Results

```
               precision    recall  f1-score   support

        Benign     0.95      0.96      0.96     9711
       DoS_Hulk     0.93      0.91      0.92     2315
   Probe_PortScan   0.89      0.88      0.88     1337
          …

    Macro‑F1 ≈ 0.92
```

A full HTML report with confusion matrices is generated under `reports/latest/` after each training run.

---


## Contributing

1. **Fork** the repo & create your feature branch (`git checkout -b feat/my-feature`).
2. **Commit** your changes (`git commit -m 'feat: add my feature'`).
3. **Push** to the branch (`git push origin feat/my-feature`).
4. **Open a Pull Request** explaining *why* and *how*.

Please run `pytest` before submitting.\
See [`CONTRIBUTING.md`](CONTRIBUTING.md) for coding style & CI details.

---

## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.
