# Leveraging LoRA for Efficient Fine-Tuning of Large Models in NLP

This repository contains the code and resources used for the report **"Leveraging LoRA for Efficient Fine-Tuning of Large Models in NLP"**, which explores the use of **Low-Rank Adaptation (LoRA)** for fine-tuning large pretrained language models (LLMs) such as **T5-Small**, **BART-Base**, and **PEGASUS**. The report focuses on two key NLP tasks: **document classification** and **text summarization**, demonstrating the effectiveness of LoRA in balancing computational efficiency and task performance.

## Features

- Implementation of LoRA fine-tuning for T5, BART, and PEGASUS models using Hugging Face Transformers.
- Tasks:
  - **Text summarization**: CNN/Daily Mail, Gigaword datasets.
  - **Document classification**: SST-2, AG News datasets.
- Evaluation metrics:
  - ROUGE-1, ROUGE-2, ROUGE-L for summarization tasks.
  - Accuracy and F1-score for classification tasks.
- Results highlight LoRA’s efficiency in reducing computational cost while maintaining competitive performance.

## Results

Key findings from the experiments:

- **PEGASUS** outperformed other models in summarization tasks with the highest ROUGE scores.
- **BART-Base** achieved the best results in sentiment classification (SST-2 dataset).
- **T5-Small** demonstrated superior performance in multi-class document classification (AG News dataset) with minimal computational overhead.

## Setup

### Requirements

- Python 3.8+
- PyTorch 1.13+
- Hugging Face Transformers

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/shreeman1000/DLNLP_term_paper.git
   cd Efficient_fine_tuning_LLMs

