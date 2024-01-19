# AlphAI

AlphAI is a high-level open-source Python toolkit designed for efficient AI development and in-depth GPU profiling. Supporting popular tensor libraries like [PyTorch](https://pytorch.org/get-started/locally/) and [Jax](https://github.com/google/jax), it optimizes developer operations on GPU servers and integrates seamlessly with [American Data Science Labs](https://dashboard.amdatascience.com), offering robust control over remote Jupyter Lab servers and environment runtimes.

## Features

- **GPU Profiling and Analytics**: Advanced GPU profiling capabilities to maximize resource efficiency and performance.
- **Benchmarking Tools**: Pythonic, easy-to-use tools for evaluating and comparing model performance.
- **Remote Jupyter Lab Integration**: Programmatic management of remote Jupyter Lab servers for enhanced productivity.
- **Local Tensor Model Support**: Streamlines the integration and management of tensor models from providers like Hugging Face.
- **Tensor Engine Compatibility**: Fully compatible with PyTorch, with upcoming support for Jax and TensorFlow.

## Quick Start

### Installation

Install AlphAI easily using pip:

```bash
pip install alphai

# If you'd like to install torch in a Linux machine with CUDA-drivers
pip install alphai[torch]
```

### Authentication Pre-requisites

Although not strictly required to use the computational functions of the alphai package, it is recommended to create an account at [American Data Science](https://dashboard.amdatascience.com) and generate an API key to make use of your two free remote Jupyter Lab servers.

You don't need an API key to use the GPU profiling, benchmarking, and generate modules.


### Basic Usage

Here's a quick example to get started with AlphAI:

```python
from alphai import AlphAI

# Initialize AlphAI

aai = AlphAI(
  api_key=os.environ.get("ALPHAI_API_KEY"),
)

# Start remote Jupyter Lab servers
aai.start_server()

# Upload to your server's file system 
aai.upload("./main.py")

# Start python kernel and run code remotely
code = "print('Hello world!')"
aai.run_code(code)

```

## Documentation and Detailed Usage

For more documentation and detailed instructions on how to use AlphAI's various features, please refer to our [Documentation](https://alphai.amdatascience.com).

### [Working with Tensor Models](https://alphai.amdatascience.com/americandatascience/alphai/models/hugging-face/)

Guidance on integrating and leveraging tensor models.

### [GPU Profiling and Analytics](https://alphai.amdatascience.com/americandatascience/alphai/gpu-profiling/)

Comprehensive features for GPU profiling and analytics.

### [Integration with American Data Science Labs](https://alphai.amdatascience.com/americandatascience/alphai/servers/)

Discover the benefits of integrating AlphAI with American Data Science Labs.

## System Requirements

- Python 3.9+
- PyTorch (recommnended) or Jax (limited support)
- Linux OS i.e. Ubuntu 18.04+

## Contributing

We welcome contributions! Please see our [Contribution Guidelines](https://github.com/americandatascience/alphai/README.md) for more information.

## License

AlphAI is released under the [Apache 2.0](https://github.com/americandatascience/alphai/LICENSE.txt) license.

## Support and Contact

For support or inquiries about enterprise solutions, contact us at [info@amdatascience.com](mailto:info@amdatascience.com).
