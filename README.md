# Mistral-7B Local Inference

A minimal implementation of the Mistral-7B model for offline use. This implementation runs completely locally on your CPU and doesn't require internet connection after initial setup.

## Features

- Uses Mistral-7B model (4GB GGUF format)
- Runs completely offline
- Optimized for CPU usage
- Handles factual Q&A well
- Low memory footprint

## Setup

1. Install the requirements:
```bash
pip install -r requirements.txt
```

2. Download a compatible Mistral-7B model file from Hugging Face. Place it in the `models` directory.

## Usage

Simply run:
```bash
python inference.py
```

Then type your question when prompted.

## Requirements
- Python 3.8+
- llama-cpp-python
- numpy

## Note
Make sure you have enough disk space for the model file (typically 4-5GB for a quantized 7B parameter model).
