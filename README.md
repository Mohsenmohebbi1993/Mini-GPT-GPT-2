# MiniGPT Pre-Training from Scratch with PyTorch

A lightweight, end-to-end implementation of a GPT-style autoregressive language model (Decoder-Only Transformer) and its complete training pipeline from scratch using Python and PyTorch.

---

## 📌 Project Overview

The objective of this project is to implement and pre-train MiniGPT from scratch using PyTorch. The model architecture closely mirrors GPT-2, scaled down and simplified for educational purposes to provide an in-depth understanding of the internal mechanisms of Large Language Models (LLMs).

1. **Custom Tokenizer Implementation (Byte-level BPE)**
2. **Pre-Training Data Pipeline Engineering**
3. **Transformer Decoder Architecture from Scratch**
4. **Pre-Training & Autoregressive Generation**

---

## 🏗️ Architecture & Technical Specifications

- **Model Type:** Decoder-Only Transformer (GPT-2 style)
- **Framework:** PyTorch (Pure Tensor Operations & `nn.Module`)
- **Components:**
  - Token & Learnable Positional Embeddings
  - Multi-Head Causal Self-Attention with Attention Masking
  - Pre-Layer Normalization (Pre-LN) & Residual Connections
  - Position-wise Feed-Forward Networks (MLP / GeLU)
  - Vocabulary Projection Head & Cross-Entropy Loss
  - Autoregressive Decoding (Greedy / Top-$k$ / Top-$p$ sampling)

---

---

## Model Architecture (Decoder-Only Transformer)
The architecture follows a GPT-2 style Decoder-Only Transformer containing:
- Token Embedding
- Positional Embedding
- Multi-Head Self-Attention
- Causal Mask
- Residual Connections
- Layer Normalization (Pre-LN)
- Feed-Forward Network (MLP)
- Final LayerNorm
- Linear Projection to Vocabulary Space
- Cross-Entropy Loss
- Autoregressive Text Generation

---

## Deliberately Simplified Features
The following features are omitted for educational simplicity:
- Learning Rate Scheduler
- Dropout
- Mixed Precision
- Flash Attention
- KV Cache
- Rotary Position Embedding (RoPE)
- RMSNorm
- SwiGLU
- Parallel Residual
- Grouped Query Attention (GQA)
- Large-scale training

---

## Implementation Order & File Structure
1. `tokenizer1.py` (Mandatory)
2. `tokenizer2.py` (Mandatory)
3. `data_pipeline.py` (Mandatory)
4. `mini_gpt.py` / `torch_gpt_mini.py` (Bonus)

---

## Bonus Grading
- **Mandatory Tasks:** Completing the first three files (`tokenizer1.py`, `tokenizer2.py`, `data_pipeline.py`).
- **Bonus 1 (+20%):** Implementing `mini_gpt.py` and successfully overfitting the model on the provided corpus (demonstrating loss reduction and basic autoregressive word generation).
- **Bonus 2 (+30%):** Integrating the data pipeline built in Step 3 into `mini_gpt.py` using the previously extracted doctor comments dataset.

## 📁 Repository Structure
```text
.
├── tokenizer1/          # Theoretical foundation of Tokenization (BPE, WordPiece, SentencePiece)
├── tokenizer2/          # Implementation of Custom Byte-Pair Encoding (BPE) Tokenizer
│   ├── pre_tokenizer
│   ├── bpe_trainer
│   └── tokenizer_engine
├── data_pipeline/       # End-to-end Pre-training Data Pipeline
│   ├── cleaning & normalization
│   ├── MinHash deduplication
│   └── sequence packing & batching
└── mini_gpt/            # Transformer Model & Training Engine
├── model.py         # Multi-Head Attention & Transformer Blocks
├── train.py         # Pre-training loop, optimizer, and loss tracking
└── generate.py      # Autoregressive text generation
