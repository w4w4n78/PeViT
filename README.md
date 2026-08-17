# Periodic Shifted Patch on Vision Transformer for Electroencephalogram Signal-Based Emotion Recognition

[![Python 3.10.20](https://img.shields.io/badge/python-3.10.x-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7.x](https://img.shields.io/badge/pytorch-2.7.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official PyTorch implementation of the scientific manuscript entitled "Periodic Shifted Patch on Vision Transformer for Electroencephalogram Signal-Based Emotion Recognition" submitted to *Biomedical Signal Processing and Control*.

<!-- > **Periodic Shifted Patch on Vision Transformer for Electroencephalogram Signal-Based Emotion Recognition**  
> Wawan Yunanto<sup>a</sup>, Hsing-Kuo Pao<sup>a,*</sup>, Adhi Dharma Wibawa<sup>b</sup>, Yennun Huang<sup>c</sup>, and Mohammad Iqbal<sup>b</sup>  
> <sup>a</sup> *National Taiwan University of Science and Technology, Taipei City, Taiwan*  
> <sup>b</sup> *Institut Teknologi Sepuluh Nopember, Surabaya, Indonesia*  
> <sup>c</sup> *Academia Sinica, Taipei City, Taiwan*  
> <sup>*</sup> Corresponding author: `pao@mail.ntust.edu.tw` -->

---

## Dataset

<!-- ## Abstract -->

<!-- Recognizing emotions from electroencephalogram (EEG) signals is vital for applications ranging from clinical diagnostics to human-computer interaction. Despite extensive research, current methods often rely on expensive handcrafted features or computationally intensive signal-domain transformations that struggle with the inherent noise and non-stationarity of EEG data. Furthermore, achieving high accuracy on limited datasets remains a challenge due to the overfitting tendencies of deep learning models.  -->

<!-- To address these issues, we propose **PeViT**, an efficient and effective model designed for raw EEG-based emotion recognition. We introduce a **Periodic Shifted Patch (PSP)** mechanism that provides on-the-fly data augmentation while preserving critical spatio-temporal relationships without the need for prior signal processing. This is integrated with a modified Vision Transformer architecture tailored to capture global contextual relations and long-term dependencies in EEG signals. We validated PeViT on four datasets, including the recent and large-scale **HBUED** (50 participants), alongside **DEAP**, **DREAMER**, and **SEED**. The results demonstrate that PeViT scales effectively to larger cohorts, achieving over **90% accuracy on HBUED** while maintaining state-of-the-art performance and a **30% speedup** on established benchmarks with the lowest parameter count among compared models. -->

We validated our study on four datasets, including the recent and large-scale **HBUED** (50 participants), alongside **DEAP**, **DREAMER**, and **SEED**. Each dataset is widely available through their official channel, however, they are behind a license agreement that users need to sign and agree to the terms and conditions before accessing the dataset. 

- HBUED: https://github.com/tensorground/HBUED.github.io 
- DEAP: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html
- DREAMER: https://zenodo.org/records/546113
- SEED: https://bcmi.sjtu.edu.cn/home/seed/index.html 

The preprocessing for the later 3 datasets are provided by TorchEEG, and can be use right away. However, the preprocessing for HBUED dataset is not provided by TorchEEG, and need to be preprocessed manually. In this repository, we have provided the preprocessing script alongside our propose method for the HBUED dataset.

---

<!-- ## Architecture Overview

```
                      ┌──────────────────────────────────────┐
                      │        Emotion Prediction            │
                      └──────────────────▲───────────────────┘
                                         │
                              ┌──────────┴──────────┐
                              │  Linear Classifier  │
                              └──────────▲──────────┘
                                         │
                              ┌──────────┴──────────┐
                              │ Transformer Encoder │
                              │   (MSA + MLP + LN)  │
                              └──────────▲──────────┘
                                         │
                  ┌──────────────────────┴──────────────────────┐
                  │ Positional Embedding + Class Token Embedding│
                  └──────────────────────▲──────────────────────┘
                                         │
                          ┌──────────────┴──────────────┐
                          │  Linear Patch Projection    │
                          └──────────────▲──────────────┘
                                         │
            ┌────────────────────────────┴────────────────────────────┐
            │       Periodic Shifted Patch (PSP) Augmentation          │
            │ (Overlapping shifted windows w & temporal patches p_s)  │
            └────────────────────────────▲────────────────────────────┘
                                         │
                             Raw Multi-Channel EEG Signals
```

### Key Contributions
1. **Periodic Shifted Patch (PSP) Mechanism**: Lightweight, on-the-fly EEG data augmentation using overlapping temporal shifted windows that inherently preserves cortical spatio-temporal relationships and addresses EEG signal non-stationarity.
2. **Raw Time-Domain Processing**: Eliminates the need for expensive frequency-domain transformations (FFT/STFT), graph construction (GNNs), or spatial projections—achieving a **30% speedup** and state-of-the-art parameter efficiency.
3. **Cross-Subject Robustness**: Rigorous 60:20:20 subject-independent holdout evaluation across 4 benchmark datasets (DEAP, DREAMER, SEED, and HBUED). -->

---

## Repository Structure

```
PeViT/
├── configs/                 # Dataset-specific hyperparameter configurations
│   ├── deap.yaml
│   ├── dreamer.yaml
│   ├── seed.yaml
│   └── hbued.yaml
├── datasets/                # EEG PyTorch Dataset handlers & synthetic data generator
│   ├── __init__.py
│   ├── base_dataset.py      # EEGDataset, holdout split, and gender-split utilities
│   └── mock_data.py         # Synthetic EEG data generator for fast verification
├── models/                  # PyTorch model definitions
│   ├── __init__.py
│   ├── psp_encoder.py       # Periodic Shifted Patch (PSP) encoder (Algorithm 1)
│   └── pevit.py             # PeViT Vision Transformer architecture
├── scripts/                 # CLI entry points
│   ├── train.py             # Model training script
│   └── evaluate.py          # Model evaluation script
├── utils/                   # Metrics and trainer implementation
│   ├── __init__.py
│   ├── metrics.py           # Accuracy, F1-Score, Precision, Recall
│   └── trainer.py           # Training and evaluation engine
├── citation.bib             # BibTeX citation file
├── LICENSE                  # MIT License
├── README.md                # Paper repository documentation
└── requirements.txt         # Project dependencies
```

<!-- ---

## ⚙️ Hyperparameters

Summarized hyperparameter settings across the four evaluated datasets (from Manuscript Table 3):

| Category | Parameter | DEAP | DREAMER | SEED | HBUED |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **PSP Mechanism** | Shifted Window ($w$) | 16 | 16 | 20 | 20 |
| | Patch Size ($p_s$) | 32 | 32 | 40 | 40 |
| **PeViT Architecture** | Latent Dimension ($\ell$) | 128 | 128 | 200 | 200 |
| | Encoder Layers ($o$) | 3 | 3 | 6 | 3 |
| | Attention Heads | 4 | 4 | 8 | 4 |
| | MLP Dimension | 32 | 64 | 64 | 32 |
| **Training Setting** | Optimizer | Adam | Adam | Adam | Adam |
| | Learning Rate | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| | Epochs | 300 | 10000 | 700 | 1000 |

---

## 📊 Experimental Results

### 1. General Accuracy Comparison with State-of-the-Art (Table 4)

| Method | DEAP (Arousal) | DEAP (Valence) | DREAMER (Arousal) | DREAMER (Valence) | SEED | HBUED (Arousal) | HBUED (Valence) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CCNN [35] | 90.24% | 89.45% | - | - | - | - | - |
| FBCCNN [22] | 90.26% | 88.90% | - | - | - | - | - |
| MTCNN [25] | 96.62% | 96.28% | - | - | - | - | - |
| Balic et al. [2] | 92.17% | 94.64% | - | - | - | - | - |
| GANSER [37] | 94.21% | 93.52% | 84.84% | 85.28% | - | - | - |
| ACRNN [30] | 93.38% | 93.72% | 97.98% | **97.93%** | - | - | - |
| MLF-CapsNet [21] | 98.31% | 97.97% | 95.26% | 94.59% | - | - | - |
| Ramzan et al. [24] | 97.39% | 97.41% | - | - | 93.74% | - | - |
| Shao et al. [28] | - | - | - | - | 85.80% | - | - |
| Fdez et al. [7] | - | - | - | - | 79.60% | - | - |
| Topic et al. [31] | - | - | - | - | 88.45% | - | - |
| Yuvaraj et al. [36] | 79.90% | 78.18% | 91.60% | 87.59% | 96.50% | - | - |
| **PeViT (Ours)** | **98.67%** | **98.57%** | **98.40%** | 97.28% | **98.49%** | **94.69%** | **95.30%** |

### 2. Gender-Based Emotion Recognition Accuracy (Table 5)

| Dataset | Male (Arousal) | Male (Valence) | Female (Arousal) | Female (Valence) |
| :--- | :---: | :---: | :---: | :---: |
| **DEAP** | 99.17% | 99.18% | 99.02% | 98.63% |
| **DREAMER** | 98.81% | 98.04% | 99.40% | 97.95% |
| **SEED** | - | 98.85% | - | 99.69% |

### 3. Model Parameters & Complexity Comparison (Table 6)

| Model | DEAP | DREAMER | SEED |
| :--- | :---: | :---: | :---: |
| CCNN [35] | 5,909,728 | - | - |
| MTCNN [25] | 1,216,004 | - | - |
| Balic et al. [2] | 941,106 | - | - |
| GANSER [37] | 2,671,250 | 2,671,250 | - |
| ACRNN [30] | 402,009 | 401,451 | - |
| MLF-CapsNet [21] | 110,850,000 | 44,150,000 | - |
| Ramzan et al. [24] | 2,850,841 | - | 2,850,841 |
| **PeViT (Ours)** | **356,066** | **251,906** | **1,995,267** |

--- -->

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/w4w4n78/PEViT.git
   cd PEViT
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Training

To train PeViT on synthetic mock EEG datasets (or your preprocessed EEG data):

```bash
# Train on DEAP dataset configuration
python scripts/train.py --config configs/deap.yaml --dataset DEAP --epochs 50

# Train on DREAMER dataset configuration
python scripts/train.py --config configs/dreamer.yaml --dataset DREAMER --epochs 50

# Train with gender-stratified split (Male subjects only)
python scripts/train.py --config configs/deap.yaml --dataset DEAP --gender M
```

### Evaluation

To evaluate a saved PeViT model checkpoint:

```bash
python scripts/evaluate.py --checkpoint pevit_checkpoint.pth --dataset DEAP
```

---

## Citation

If you find PeViT useful in your research, please cite our paper:

```bibtex
@article{yunanto2026pevit,
  title={Periodic Shifted Patch on Vision Transformer for Electroencephalogram Signal-Based Emotion Recognition},
  author={Yunanto, Wawan and Pao, Hsing-Kuo and Wibawa, Adhi Dharma and Huang, Yennun and Iqbal, Mohammad},
  journal={Biomedical Signal Processing and Control},
  year={2026},
  publisher={Elsevier}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
