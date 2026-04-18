
# Char-Level-LM-Comparison: LSTM vs. Mini-Transformer

A production-grade, fully containerized PyTorch implementation comparing Recurrent Neural Networks (LSTM) and Self-Attention mechanisms (Mini-Transformer) built from scratch for character-level text generation. 

This repository serves as a rigorous academic benchmark, trained on the classic Tiny Shakespeare dataset, to evaluate the parameter efficiency, training stability, and generative capabilities of both architectures strictly within CPU-bound environments.

---

## 📖 Project Overview

Modern Large Language Models (LLMs) rely heavily on Transformer architectures, which have largely superseded RNNs/LSTMs in NLP tasks. This project strips away the abstractions of modern libraries (like Hugging Face) to implement the core mathematical components of both models from scratch using `torch.nn.Module`. 

**Key Objectives Achieved:**
* **From-Scratch Architecture:** Implementation of Multi-Head Causal Self-Attention, Positional Encoding, and LSTM cell linear projections without relying on pre-packaged `nn.Transformer` modules.
* **Production Engineering:** 100% containerized execution using Docker and `docker-compose`, ensuring reproducible environments without local dependency conflicts.
* **Optimization Best Practices:** Custom batching generators, gradient clipping to prevent NaN losses, and hyperparameter management via `.env` configurations.
* **Autoregressive Generation:** Implementation of temperature scaling to control the deterministic vs. creative nature of the generated text.

---

## 🌍 Real-World Applications & Problems Solved

While this project uses a character-level Shakespeare dataset for demonstration, the underlying mathematical architectures solve critical industry problems:

1. **The Context Bottleneck (Solved by Transformers):** * *The Problem:* LSTMs compress historical sequence data into a fixed-size hidden state, which acts as an information bottleneck and causes "forgetting" over long sequences.
   * *The Solution:* The Transformer's Self-Attention mechanism looks at the entire sequence simultaneously, calculating $QK^T$ affinities. This is the exact foundational mechanism behind models like GPT-4 and BERT, used for document summarization, semantic search, and complex code generation.
2. **Time-Series & Sequential Forecasting (Solved by LSTMs):** * *The Problem:* Not all data has a fixed context window. Stock market feeds, IoT sensor telemetry, and server anomaly logs arrive continuously.
   * *The Solution:* Because LSTMs process data sequentially via hidden states, they remain the industry standard for lightweight, low-latency edge computing tasks like predictive maintenance and financial volatility forecasting.
3. **Controlled Generation via Temperature Scaling:** * By manipulating the logits prior to the Softmax distribution, this project demonstrates how AI companies tune models for different use cases—low temperature (0.1 - 0.5) for strict, factual RAG (Retrieval-Augmented Generation) pipelines, and high temperature (1.0+) for creative writing assistants.

---

## 📂 Repository Structure

```text
/
├── input/                  # Mount point for raw datasets
│   └── tinyshakespeare.txt
├── results/                # Mount point for generated models and artifacts
│   ├── lstm_model.pth
│   ├── transformer_model.pth
│   ├── loss_curves.png
│   ├── generated_samples.json
│   └── comparison_report.md
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── prepare_data.py     # Tokenization and tensor creation
│   ├── model_lstm.py       # PyTorch LSTM architecture
│   ├── model_transformer.py# PyTorch Self-Attention architecture
│   ├── train.py            # Unified training engine
│   ├── generate.py         # Autoregressive inference with temperature
│   └── evaluate.py         # Master script for artifact generation
├── .env.example            # Hyperparameter documentation
├── docker-compose.yml      # Volume and service orchestration
├── Dockerfile              # CPU-optimized Python environment
└── README.md
```

---

## 🚀 Execution Guide

This project requires zero local setup other than Docker. All dependencies and executions are isolated within the container.

### 1. Build the Environment
```bash
docker-compose build
```

### 2. Prepare the Data
Reads the raw text, builds a character-level vocabulary, and saves PyTorch `.pt` tensors to `/results`.
```bash
docker-compose run --rm app python src/prepare_data.py
```

### 3. Train the Models
Train both architectures independently. Checkpoints and loss histories are saved automatically.
```bash
docker-compose run --rm app python src/train.py --model lstm
docker-compose run --rm app python src/train.py --model transformer
```

### 4. Custom Text Generation (Inference)
Generate text interactively by adjusting the seed phrase and temperature.
```bash
docker-compose run --rm app python src/generate.py --model transformer --model_path results/transformer_model.pth --seed_text "ROMEO:" --temperature 1.0
```

### 5. Generate Final Evaluation Artifacts
Run the master evaluation suite to calculate Perplexity, output JSON samples, and plot loss curves.
```bash
docker-compose run --rm app python src/evaluate.py
```

---

## 👨‍💻 Author

**Billakurti Venkata Suryanarayana**
* **Roll No:** 23MH1A4409**
