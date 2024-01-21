# Getting Started

AlphAI is a powerful and high-level tool for running analytics, LLMs, and profiling on GPU servers. It also provides you with a client to American Data Science Labs for powerful control over your remote Jupyter Lab servers and environment runtimes.

If you're familiar with python, you can install AlphAI with pip, the python package manager.

## Installation

### with pip

Install the python package by running `pip install alphai`.

To run profiling and model inference on GPUs with alphai, an installation of PyTorch in a Linux OS with CUDA-enabled is required.

Benchmarking, American Data Science client, and other model utilities will work without a GPU, Linux OS, or PyTorch installed.

If your CUDA drivers are up to date, you can install alphai with GPU-enabled torch by running `pip install alphai[torch]`.

## quickstart

You can check if AlphAI was successfully installed by trying out the benchmarking tools:

```python
from alphai import AlphAI

aai = AlphAI()

def some_function(x, y):
    return x+y

aai.start_timer()
some_function(1, 2)
aai.stop_timer()

aai.benchmark(some_function, 1, 2, num_iter = 100)
```

If you have torch installed, you can even use the `generate()` feature.

```python
prompt = "Hello there!"

aai.start_timer()
aai.generate(prompt)
aai.stop_timer()

aai.benchmark(aai.generate, prompt, num_iter = 5)
```