# MiniGPT Pre-Training from Scratch with PyTorch

A lightweight, end-to-end implementation of a GPT-style autoregressive language model (Decoder-Only Transformer) and its complete training pipeline from scratch using Python and PyTorch.

---

## 📌 Project Overview

This project demonstrates the core engineering and theoretical principles behind modern Large Language Models (LLMs). It covers the full lifecycle of building an LLM without external high-level abstraction libraries (e.g., Hugging Face `transformers`), spanning:

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
