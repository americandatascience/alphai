# Getting Started

AlphAI is a powerful and high-level tool for running analytics, LLMs, and profiling on GPU servers. It also provides you with a client to American Data Science Labs for powerful control over your remote Jupyter Lab servers and environment runtimes.

If you're familiar with python, you can install AlphAI with pip, the python package manager.

## Installation

### with pip

Install the python package by running `pip install alphai`.

To run profiling and model inference on GPUs with alphai, an installation of PyTorch in a Linux OS with CUDA-enabled is required.

Benchmarking, American Data Science client, and other model utilities will work without a GPU, Linux OS, or PyTorch installed.

If your CUDA drivers are up to date, you can install alphai with GPU-enabled torch by running `pip install alphai[torch]`.


### Authentication Pre-requisites

Although not strictly required to use the computational functions of the alphai package, it is recommended to create an account at [American Data Science](https://dashboard.amdatascience.com) and generate an API key to make use of your two free remote Jupyter Lab servers.

You don't need an API key to use the GPU profiling, benchmarking, and generate modules.

## American Data Science Labs

If you're authenticated with your AmDS account and api key, you can start and stop your servers.


```python
import os
from alphai import AlphAI

aai = AlphAI(
    api_key=os.environ.get("ALPHAI_API_KEY"),
)

# Starting default server
# May take a moment to get ready
aai.start_server()

# Upload to your server's file system 
aai.upload("./main.py")

# Start python kernel and run code remotely
code = "print('Hello world!')"
aai.run_code(code)

```


To stop the servers run below:


```python
aai.stop_server()
```

