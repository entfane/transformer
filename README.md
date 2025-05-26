# Transformer Decoder Language Model 🚀

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

A lightweight yet powerful Transformer Decoder implementation (~10M parameters) for language modeling tasks. This repository contains a clean, well-structured implementation of a transformer decoder with training and text generation capabilities.

## 🔥 Key Features

- Pure PyTorch implementation
- Clean, modular architecture
- Training and inference scripts
- GPU/CPU support
- Validation during training
- Vocabulary management

## 🏗️ Architecture

The model follows the standard Transformer Decoder architecture with:

- Multi-head self-attention
- Position-wise feed-forward networks
- Layer normalization
- Residual connections
- Learned positional embeddings

With ~10 million parameters, it's lightweight enough for experimentation while being powerful enough to learn meaningful language patterns.

## ⚙️ Setup

1. Clone the repository:
```bash
git clone https://github.com/entfane/transformer.git
cd transformer-decoder
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚂 Training

First provide the corpus as a .txt file.

Train the model with default parameters:
```bash
python train.py
```

Customize training with command-line arguments:
```bash
python train.py \
    --lr 3e-4 \
    --iter 1000 \
    --batch_size 64 \
    --token_to_idx my_vocab.pkl \
    --corpus my_text.txt \
    --save_path best_model.pt \
    --device cuda \
    --validation 0.2 \
    --validation_iter 20 \
    --validation_interval 50
```

For more details regarding training arguments:
```bash
python train.py --help
```

### Training Arguments

| Argument | Description | Default Value |
|----------|-------------|---------------|
| `--lr` | Learning rate | 4e-4 |
| `--iter` | Training iterations | 10 |
| `--batch_size` | Batch size | 1 |
| `--token_to_idx` | Path to vocabulary pickle file | "token_to_idx.pkl" |
| `--corpus` | Path to training corpus (.txt) | "corpus.txt" |
| `--save_path` | Model save path (.pt/.pth) | "model.pt" |
| `--device` | Device to use (cpu/cuda) | auto-detected |
| `--validation` | Validation set percentage | 0.1 |
| `--validation_iter` | Validation iterations | 10 |
| `--validation_interval` | Validate every N iterations | 10 |

## 🎲 Text Generation

Generate text with your trained model:
```bash
python text_generation.py
```

For successful generation a vocabulary of idx_to_token.pkl should exist.

## 📂 Project Structure

```
.
├── constants.py
├── download_raw.py     # Script to download tiny Shakespear
├── model.py             # Transformer Decoder implementation
├── tests.py             # Various tests covering separate Transformer architecture components
├── text_generation.py    # Text generation script
├── tools.py             # Helper functions
├── train.py              # Training script
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## 📜 License

MIT

---