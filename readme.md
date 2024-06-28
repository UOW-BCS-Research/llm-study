# LLM Study: Exploring Large Language Model Embeddings
Welcome to the UOW-BCS-Research/llm-study repository! This project aims to investigate and experiment with Large Language Model (LLM) embeddings. Whether you’re a researcher, developer, or curious learner, this repository provides a playground for LLM-related concepts.

## What Is LLM?
1. Large Language Models (LLMs): These are powerful neural network-based models that learn to generate and understand natural language. Examples include GPT-4, and Llama3.
2. Embeddings: LLMs create dense vector representations (embeddings) for words, sentences, or documents. These embeddings capture semantic meaning and context.
## Repository Contents
  - Explore LLM embeddings using various pre-trained models (e.g., Hugging Face Transformers, Sentence Transformers).
  - Test hypotheses, fine-tune embeddings, and experiment with different tasks.

## Get Started
1. Install Dependencies: Ensure you have the necessary Python libraries installed. Consider using MongoDB, a vector database, for efficient storage and retrieval.
2. Start the playground
```
streamlit run src/app_hf.py --server.port=8501 --server.address=0.0.0.0
streamlit run src/app_ollama.py --server.port=8501 --server.address=0.0.0.0
```
