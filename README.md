# Sem-HAR: Semantic Self-Supervised Learning for Layout-Agnostic Human Activity Recognition

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-ee4c2c.svg)](https://pytorch.org/)
[![Ray Tune](https://img.shields.io/badge/Ray%20Tune-Distributed%20Tuning-028cf0.svg)](https://docs.ray.io/en/latest/tune/index.html)

**Author:** Aayush Atul Chandak | **Institution:** Georgia Institute of Technology | **Course:** CS 8903
**Mentor:** Sourish Dhekane | **Faculty PI:** Dr. Thomas Ploetz

---

## 📌 Overview
Wearable Inertial Measurement Unit (IMU) models for Human Activity Recognition (HAR) frequently suffer from severe performance degradation when subjected to domain shifts, particularly changes in sensor placement. Traditional self-supervised learning (SSL) methods rely on signal-level augmentations that fail to capture the underlying physical meaning of movements. 

**Sem-HAR** is a novel multimodal framework that replaces spatial and device-specific semantics with motion-level semantics. By aligning continuous raw IMU sensor waves with Large Language Model (LLM)-generated textual descriptions of biomechanics, we establish a layout-agnostic representation space capable of robust cross-domain transfer.

## ✨ Key Contributions
1. **Motion-Level Semantics:** Replaces arbitrary class IDs with rich, LLM-generated natural language descriptions of motion dynamics (e.g., periodicity, intensity, and axis dominance).
2. **Discrete Biomechanical Isolation:** Upgrades standard continuous latent spaces by implementing a Vector-Quantized Contrastive Predictive Coding (VQ-CPC) bottleneck. This maps noisy continuous signals to a finite codebook of 512 discrete "motion phonemes," inherently filtering stochastic sensor noise.
3. **Automated Visual Diagnostics:** An engineered multimodal interpretability suite capable of visually fingerprinting activity distributions (Semantic Density) and mathematically isolating anomalous "divergent codes". 
4. **VLM Integration:** Extracts raw 3x3 waveform grids for anomalous codes, allowing Vision-Language Models (VLMs) to directly observe and verify structural physical realities (e.g., impact forces, periodicity).

## 🧠 Architecture
### The Dual-Tower Framework
* **Text Tower:** A frozen `DistilBERT` model used to encode LLM-generated biomechanical descriptions (Teacher).
* **IMU Tower:** A trainable 1D-CNN designed to extract hierarchical features from raw multiaxial IMU signals (Student).

### The VQ-CPC Upgrade & Hybrid Objective
To overcome feature smearing caused by IMU drift and noise, the 1D-CNN output is forced through a discrete VQ bottleneck. The network is trained using a hybrid objective function:
* **$L_{total} = L_{contrastive} + \beta \cdot L_{VQ}$**
* $L_{contrastive}$ (InfoNCE) pulls matching IMU-Text pairs together in the shared latent space.
* $L_{VQ}$ (Commitment Loss) uses a straight-through estimator to stabilize the dictionary vectors and force the encoder to commit to the learned phonemes.

## 📊 Results: Downstream Transfer Learning
A Multi-Layer Perceptron (MLP) classification head was attached directly to the pre-trained IMU backbone, bypassing the text tower entirely for linear probing evaluations. The VQ-CPC architecture yielded highly robust, noise-resistant features.

| Encoder Architecture | Classification Head | Validation Accuracy | Macro F1-Score |
| :--- | :--- | :--- | :--- |
| Continuous 1D-CNN (Baseline) | Deep MLP | 69.70% | 71.30% |
| **VQ-CPC 1D-CNN (Ours)** | **Deep MLP** | **77.34%** | **77.01%** |

## 🚀 Distributed Optimization
The training loop and execution scripts are deeply integrated with the **Ray Tune** framework to enable distributed hyperparameter sweeps across high-performance GPU compute nodes. 
* **Engineering Fix:** Implemented custom PyTorch DataLoader routing to properly unpack the multi-variable tuple from the VQ layer (quantized features, VQ commitment loss, discrete indices, perplexity) to compute the final backward pass without `ValueError` execution crashes.

## 📂 Dataset Preprocessing
The framework was evaluated using the **Capture-24** dataset (unlabelled, "in-the-wild" wrist-worn accelerometer data). 
* Preprocessing enforces strict float32/uint8 data typing and memory-optimized programmatic garbage collection to prevent OOM errors. 
* Time-series data is segmented into fixed-length windows of 40 timesteps.

## 🔮 Future Work
* **Proactive Agentic AI:** Utilizing the 512-word motion codebook to allow autonomous AI agents to reason over a chronological sequence of human kinematics for complex medical interventions rather than simple binary thresholding.
* **Edge Deployment:** Leveraging compiler engineering to optimize the discrete VQ representations for ultra-low-power microcontrollers.

## 📖 Acknowledgments
This work was conducted as part of the CS 8903 curriculum at the Georgia Institute of Technology, under the mentorship of Sourish Dhekane and Dr. Thomas Ploetz.
