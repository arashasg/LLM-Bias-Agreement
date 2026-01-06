# LLM-Bias-Agreement: Quantifying Consensus in Large Language Model Bias Evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ACL Submission](https://img.shields.io/badge/ACL-2026_Submission-red.svg)](https://aclweb.org)

> **Official repository for the paper: "Quantifying Consensus: A Framework for Auditing Bias Metrics in Large Language Models"**

This repository contains the complete pipeline for auditing bias metrics across 8 benchmark datasets and 10 state-of-the-art LLMs (including Llama 3, DeepSeek-R1, and Phi-4). We introduce **Metric Agreement Score (MeAS)** and **Model Agreement Score (MoAS)** to formalize how bias benchmarks agree (or disagree) on model rankings.

<p align="center">
  <img src="assets/deepseek_bias_metric_ranks.png" width="800" alt="Heatmap showing DeepSeek vs Western Model Bias Profiles">
  <br>
  <em>Figure 1: Cross-metric correlation analysis revealing distinct bias profiles between DeepSeek models and Western open-weight models.</em>
</p>

## 📚 Table of Contents
- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Reproducing Results](#-reproducing-results)
    - [Step 1: Metric Evaluation](#step-1-metric-evaluation)
    - [Step 2: Analysis & Agreement Scores](#step-2-analysis--agreement-scores)
- [Key Findings](#-key-findings)
- [Citation](#-citation)

## 🔍 Project Overview

Despite the proliferation of bias benchmarks, it remains unclear whether these metrics measure the same underlying construct. This project addresses this gap by:
1.  **Benchmarking 10 LLMs** across **8 diverse datasets** (CrowS-Pairs, WinoBias, BOLD, RealToxicityPrompts, etc.).
2.  Implementing both **Probabilistic** (perplexity-based) and **Generation-based** (toxicity/HONEST) evaluation pipelines.
3.  Introducing the **MeAS/MoAS Framework** to quantify inter-metric consistency.
4.  Conducting a **Leave-One-Dataset-Out (LODO)** analysis to test ranking stability.

## 📂 Repository Structure

The codebase is organized to separate data loading, experimental scripts, and analysis logic.

```text
LLM-Bias-Agreement/
├── assets/                 # Visualizations and teasers for documentation
├── configs/                # Configuration files for models and datasets
├── data/                   # Dataset storage (Raw and Processed)
│   ├── raw/                # Original benchmarks (BOLD, WinoBias, etc.)
│   └── processed/          # Preprocessed JSON inputs for evaluation
├── output/                 # Generated artifacts
│   ├── figures/            # Final correlation heatmaps & clustering dendrograms
│   └── ranks/              # CSVs containing calculated MeAS and MoAS scores
├── scripts/                # Evaluation pipelines (HPC Slurm scripts & python runners)
│   ├── generation_based/   # Scripts for BOLD, RealToxicityPrompts, etc.
│   └── probabilistic/      # Scripts for CrowS-Pairs, WinoBias, etc.
└── src/                    # Core library code
    ├── analysis/           # Logic for ranking compilation
    ├── evaluation/         # Implementation of MeAS, MoAS, and LODO stability
    └── utils/              # Helper functions for data loading and logging