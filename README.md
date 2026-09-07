# Sem-HAR: Semantic Self-Supervised Learning for Layout-Agnostic Human Activity Recognition[cite: 7]

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Framework-ee4c2c.svg)](https://pytorch.org/)
[![Ray Tune](https://img.shields.io/badge/Ray%20Tune-Distributed%20Tuning-028cf0.svg)](https://docs.ray.io/en/latest/tune/index.html)

**Author:** Aayush Atul Chandak | **Institution:** Georgia Institute of Technology | **Course:** CS 8903[cite: 7]
**Mentor:** Sourish Dhekane | **Faculty PI:** Dr. Thomas Ploetz[cite: 7]

---

## 📌 Overview
Wearable Inertial Measurement Unit (IMU) models for Human Activity Recognition (HAR) frequently suffer from severe performance degradation when subjected to domain shifts, particularly changes in sensor placement[cite: 7]. Traditional self-supervised learning (SSL) methods rely on signal-level augmentations that fail to capture the underlying physical meaning of movements[cite: 7]. 

**Sem-HAR** is a novel multimodal framework that replaces spatial and device-specific semantics with motion-level semantics[cite: 7]. By aligning continuous raw IMU sensor waves with Large Language Model (LLM)-generated textual descriptions of biomechanics, we establish a layout-agnostic representation space capable of robust cross-domain transfer[cite: 7].

## ✨ Key Contributions
1. **Motion-Level Semantics:** Replaces arbitrary class IDs with rich, LLM-generated natural language descriptions of motion dynamics (e.g., periodicity, intensity, and axis dominance)[cite: 7].
2. **Discrete Biomechanical Isolation:** Upgrades standard continuous latent spaces by implementing a Vector-Quantized Contrastive Predictive Coding (VQ-CPC) bottleneck[cite: 7]. This maps noisy continuous signals to a finite codebook of 512 discrete "motion phonemes," inherently filtering stochastic sensor noise[cite: 7].
3. **Automated Visual Diagnostics:** An engineered multimodal interpretability suite capable of visually fingerprinting activity distributions (Semantic Density) and mathematically isolating anomalous "divergent codes"[cite: 7]. 
4. **VLM Integration:** Extracts raw 3x3 waveform grids for anomalous codes, allowing Vision-Language Models (VLMs) to directly observe and verify structural physical realities (e.g., impact forces, periodicity)[cite: 7].

## 🧠 Architecture
### The Dual-Tower Framework
* **Text Tower:** A frozen `DistilBERT` model used to encode LLM-generated biomechanical descriptions (Teacher)[cite: 7].
* **IMU Tower:** A trainable 1D-CNN designed to extract hierarchical features from raw multiaxial IMU signals (Student)[cite: 7].

### The VQ-CPC Upgrade & Hybrid Objective
To overcome feature smearing caused by IMU drift and noise, the 1D-CNN output is forced through a discrete VQ bottleneck[cite: 7]. The network is trained using a hybrid objective function:
* **$L_{total} = L_{contrastive} + \beta \cdot L_{VQ}$**[cite: 7]
* $L_{contrastive}$ (InfoNCE) pulls matching IMU-Text pairs together in the shared latent space[cite: 7].
* $L_{VQ}$ (Commitment Loss) uses a straight-through estimator to stabilize the dictionary vectors and force the encoder to commit to the learned phonemes[cite: 7].

## 📊 Results: Downstream Transfer Learning
A Multi-Layer Perceptron (MLP) classification head was attached directly to the pre-trained IMU backbone, bypassing the text tower entirely for linear probing evaluations[cite: 7]. The VQ-CPC architecture yielded highly robust, noise-resistant features[cite: 7].

| Encoder Architecture | Classification Head | Validation Accuracy | Macro F1-Score |
| :--- | :--- | :--- | :--- |
| Continuous 1D-CNN (Baseline)[cite: 7] | Deep MLP[cite: 7] | 69.70%[cite: 7] | 71.30%[cite: 7] |
| **VQ-CPC 1D-CNN (Ours)**[cite: 7] | **Deep MLP**[cite: 7] | **77.34%**[cite: 7] | **77.01%**[cite: 7] |

## 🚀 Distributed Optimization
The training loop and execution scripts are deeply integrated with the **Ray Tune** framework to enable distributed hyperparameter sweeps across high-performance GPU compute nodes[cite: 7]. 
* **Engineering Fix:** Implemented custom PyTorch DataLoader routing to properly unpack the multi-variable tuple from the VQ layer (quantized features, VQ commitment loss, discrete indices, perplexity) to compute the final backward pass without `ValueError` execution crashes[cite: 7].

## 📂 Dataset Preprocessing
The framework was evaluated using the **Capture-24** dataset (unlabelled, "in-the-wild" wrist-worn accelerometer data)[cite: 7]. 
* Preprocessing enforces strict float32/uint8 data typing and memory-optimized programmatic garbage collection to prevent OOM errors[cite: 7]. 
* Time-series data is segmented into fixed-length windows of 40 timesteps[cite: 7].

## 🔮 Future Work
* **Proactive Agentic AI:** Utilizing the 512-word motion codebook to allow autonomous AI agents to reason over a chronological sequence of human kinematics for complex medical interventions rather than simple binary thresholding[cite: 7].
* **Edge Deployment:** Leveraging compiler engineering to optimize the discrete VQ representations for ultra-low-power microcontrollers[cite: 7].

## 📖 Acknowledgments
This work was conducted as part of the CS 8903 curriculum at the Georgia Institute of Technology, under the mentorship of Sourish Dhekane and Dr. Thomas Ploetz[cite: 7].
