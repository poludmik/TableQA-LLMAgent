# Optimizing LLM-Powered Agents for Tabular Data Analytics 🚀

## Overview 📋

This repository contains the code and resources for my [diploma thesis](https://dspace.cvut.cz/bitstream/handle/10467/115388/F3-DP-2024-Poludin-Mikhail-Optimizing_LLM-Powered_Agents_for_Tabular_Data_Analytics_Integrating_LoRA_for_Enhanced_Quality.pdf?sequence=-1&isAllowed=y). The project explores the use of Large Language Models (LLMs) in analyzing tabular data using natural language by generation and execution of Python code. The thesis includes a comprehensive literature review, development of an LLM-based Agent program, and performance evaluations using fine-tuned and state-of-the-art models.

## Project Structure 🗂️

The project is organized as follows:

```plaintext
TableQA-LLMAgent/       # Root directory
│
├── .github/            # CI/CD workflows
│   └── workflows/
│
├── README.md           # This README file
├── main.py             # Agent usage example
├── poetry.lock         # Poetry dependency management
├── pyproject.toml      # Readable dependencies
│
├── tableqallmagent/    # Source code of the package
│   ├── __init__.py
│   ├── agent.py        # Constructor and the main interface
│   ├── code_manipulation.py # Processing generated code
│   ├── coder_llms.py   # Forward passes for coding LLMs
│   ├── llms.py         # Higher level methods for LLMs
│   ├── logger.py       # Color constants for readability
│   ├── prompts.py      # Prompting strategies and formatting
│
├── dataset/            # Multiple datasets and preprocessing
├── dist/               # PyPI versions
├── evaluation/         # LLM-as-evaluator
├── finetuning/         # LoRA training scripts and configs
├── plots/              # Directory to store generated images
└── tests/              # pytest
```

## Installation 🔧

To get started with the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/poludmik/TableQA-LLMAgent.git
    cd TableQA-LLMAgent
    ```

2. Install the dependencies:
    ```bash
    poetry install
    ```

## Usage 🚀

You can run the main script to see the basic example functionalities of the agent:

```bash
python main.py
```

## Features ✨

- **Fine-Tuning**: Fine-tuning LLMs using LoRA and QLoRA techniques.
- **Code Generation**: Generating Python code to analyze tabular data.
- **Model Evaluation**: Rigorous benchmarks for evaluating LLM Agents.
- **MLOps**: Tracking experiments using MLOps tools to ensure reproducibility.

## Results 📊

The fine-tuning experiments significantly improved the performance of the Code Llama 7B Python model from 35.3% to 60.3% on the proposed evaluation benchmark.

## Contact 📫

For any questions or feedback, please reach out to me:

Mikhail Poludin  
[michael.poludin@gmail.com](mailto:michael.poludin@gmail.com)  

## License 📜

This project is licensed under the MIT License.
