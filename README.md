# Periodic Shifted Patch on Vision Transformer for Electroencephalogram Signal-Based Emotion Recognition

[![Python 3.10.20](https://img.shields.io/badge/python-3.10.x-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7.x](https://img.shields.io/badge/pytorch-2.7.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official PyTorch implementation of the scientific manuscript entitled "Periodic Shifted Patch on Vision Transformer for Electroencephalogram Signal-Based Emotion Recognition" submitted to *Biomedical Signal Processing and Control*.

---

## Dataset

We validated our study on four datasets, including the recent and large-scale **HBUED** (50 participants), alongside **DEAP**, **DREAMER**, and **SEED**. Each dataset is widely available through their official channel, however, they are behind a license agreement that users need to sign and agree to the terms and conditions before accessing the dataset. 



The preprocessing for the later 3 datasets are provided by TorchEEG. However, the preprocessing for HBUED dataset is not provided by TorchEEG, and need to be preprocessed manually. In this repository, we have provided the preprocessing script alongside our propose method for the HBUED dataset.

---

## Repository Structure

```
PeViT/
├── datasets/                
│   ├── __init__.py
│   ├── HBUEDDataset.py      # HBUED Dataset handlers
├── scripts/                 
│   ├── HBUED.ipynb          # Overall pipeline
├── LICENSE                  # MIT License
├── README.md                # README documentation
└── requirements.txt         # Project dependencies
```

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

### Training & Evaluation

See the HBUED.ipynb notebook.


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
