import os
from typing import List, Callable, Union
import warnings
import logging
import json
import gc
import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from hta.trace_analysis import TraceAnalysis

from alphai.util import is_package_installed, extract_param_value
from alphai.profilers.configs_base import BaseProfilerConfigs
from alphai.benchmarking.benchmarker import Benchmarker
from alphai.client.client import Client


class AlphAI:
    """
    The AlphAI class provides a high-level interface for benchmarking, memory estimation,
    and interaction with remote Jupyter Lab servers. It supports various tensor-based models
    and integrates with American Data Science Labs for managing GPU resources.

    Attributes:
        output_path (str): The path where output files are stored.
        supported_backends (List[str]): List of supported tensor backends (e.g., 'torch', 'jax').
        profiler_started (bool): Flag to indicate if the profiler has started.
        server_name (str): The name of the server for remote operations.
        api_key (str): API key for authentication with remote services.
        client (Client): Client instance for interacting with remote services.
        pt_profiler (PyTorchProfiler): Profiler instance for PyTorch.
        jax_profiler (JaxProfiler): Profiler instance for JAX.
        benchmarker (Benchmarker): Benchmarker instance for performance measurements.
        model (torch.nn.Module): The loaded PyTorch model.
        model_name_or_path (str): The name or path of the model.
    """

    def __init__(
        self,
        *,
        api_key: Union[str, None] = None,
        organization: Union[str, None] = None,
        base_url: str = None,
        output_path: str = "./alphai_profiler_store",
        server_name: str = "",
        pt_profiler_configs: BaseProfilerConfigs = None,
        jax_profiler_configs: BaseProfilerConfigs = None,
        **kwargs,
    ):
        """
        Initializes the AlphAI instance with provided configurations.

        Args:
            api_key (Union[str, None]): API key for authentication. If None, will try to read from environment.
            organization (Union[str, None]): The name of the organization. If None, will try to read from environment.
            base_url (str): The base URL for remote services. If None, defaults to a predefined URL.
            output_path (str): The path where output files are stored. Defaults to './alphai_profiler_store'.
            server_name (str): The name of the server for remote operations.
            pt_profiler_configs (BaseProfilerConfigs): Configuration for the PyTorch profiler.
            jax_profiler_configs (BaseProfilerConfigs): Configuration for the JAX profiler.
        """

        self.output_path = output_path
        self.supported_backends = ["torch", "jax", "tensorflow"]
        self.profiler_started = False
        self.server_name = server_name

        # Api
        if api_key is None:
            api_key = os.environ.get("ALPHAI_API_KEY")
        if api_key is None:
            logging.info(
                "Optional: Set the API key api_key parameter init or by setting the ALPHAI_API_KEY environment variable"
            )
        self.api_key = api_key
        if api_key:
            self.client = Client(access_token=api_key)

        if organization is None:
            organization = os.environ.get("ALPHAI_ORGANIZATION_NAME")
        self.organization = organization

        if base_url is None:
            base_url = os.environ.get("ALPHAI_BASE_URL")
        if base_url is None:
            base_url = f"https://lab.amdatascience.com"
        self.base_url = base_url

        # Directory ops
        self.pt_trace_dirs = self.get_pt_traces()

        # Profilers
        self.dict_idle_time = None
        self.dict_averages = None

        if is_package_installed("torch") and not pt_profiler_configs:
            from alphai.profilers.pytorch import PyTorchProfilerConfigs, PyTorchProfiler

            pt_profiler_configs = PyTorchProfilerConfigs()
            pt_profiler_configs.dir_path = output_path
            self.pt_profiler = PyTorchProfiler(pt_profiler_configs)

        if is_package_installed("jax") and not jax_profiler_configs:
            from alphai.profilers.jax import JaxProfilerConfigs, JaxProfiler

            jax_profiler_configs = JaxProfilerConfigs()
            jax_profiler_configs.dir_path = output_path
            self.jax_profiler = JaxProfiler(jax_profiler_configs)

        # Benchmarker
        self.benchmarker = Benchmarker()

        # HF Generate
        self.model_name_or_path = None
        self.model = None

    def start(self, tensor_backend: str = None):
        """
        Starts the profiler for the specified tensor backend.

        Args:
            tensor_backend (str): The backend to use for profiling ('torch', 'jax', 'tensorflow').
                                   If None, defaults to an available backend.
        """
        # Handle if none, not installed, or unknown tensor_backend given
        # Default to torch tensorbackend or whatever's available
        if not tensor_backend:
            if is_package_installed("torch"):
                tensor_backend = "torch"
            elif is_package_installed("jax"):
                tensor_backend = "jax"
            elif is_package_installed("tensorflow"):
                tensor_backend = "tensorflow"
            else:
                warnings.warn(
                    f"Tensor framework must first be installed from a supported library: {self.supported_backends} to enable profiling."
                )
                return
        if tensor_backend not in self.supported_backends:
            warnings.warn(
                f"Tensor framework is not supported, must be one of {self.supported_backends} to enable profiling."
            )
            return
        if not is_package_installed(tensor_backend):
            warnings.warn(f"You need to install '{tensor_backend}' to start profiling")

        if tensor_backend == "torch":
            try:
                self.pt_profiler.start()
            except:
                # Try to stop hanging profiler and try again
                self.pt_profiler.stop()
                self.pt_profiler.start()
        elif tensor_backend == "jax":
            try:
                self.jax_profiler.start()
            except:
                # Try to stop hanging profiler and try again
                self.jax_profiler.stop()
                self.jax_profiler.start()
        elif tensor_backend == "tensorflow":
            pass

        self.tensor_backend = tensor_backend
        self.profiler_started = True

    def stop(self):
        """
        Stops the currently running profiler.
        """
        if not self.profiler_started or not self.tensor_backend:
            warnings.warn(f"Profiler never started")
            return

        if self.tensor_backend == "torch":
            self.pt_profiler.stop()
        elif self.tensor_backend == "jax":
            self.jax_profiler.stop()
        elif self.tensor_backend == "tensorflow":
            pass

        self.profiler_started = False

    def step(self):
        """
        Advances the profiler by one step. Mainly used for the PyTorch profiler.
        """
        self.pt_profiler.step()

    def __call__(self, tensor_backend: str = None):
        # Allows for param in context manager
        # self.tensor_backend only set with context manager or in start()
        self.tensor_backend = tensor_backend
        return self

    def __enter__(self):
        self.start(tensor_backend=self.tensor_backend)

    def __exit__(self, exc_type, exc_val, exc_t):
        self.stop()

    # API Methods
    def get_servers(self):
        """
        Retrieves the list of available servers from the remote service.

        Returns:
            A list of servers if successful, or raises an exception if the user is not authenticated.
        """
        if not self.api_key:
            raise ValueError("Requires user authentication with an API Key")
        return self.client.get_servers()

    def start_server(self, server_name: str = None):
        """
        Starts a server with the given name.

        Args:
            server_name (str): The name of the server to start. If None, uses the server name set in the instance.

        Returns:
            Response from the server start request.
        """
        if not self.api_key:
            raise ValueError("Requires user authentication with an API Key")
        # Use set self.server_name if not provided
        if server_name is None:
            server_name = self.server_name
        return self.client.start_server(server_name=server_name)

    def stop_server(self, server_name: str = None):
        """
        Stops a server with the given name.

        Args:
            server_name (str): The name of the server to stop. If None, uses the server name set in the instance.

        Returns:
            Response from the server stop request.
        """
        if not self.api_key:
            raise ValueError("Requires user authentication with an API Key")
        # Use set self.server_name if not provided
        if server_name is None:
            server_name = self.server_name
        return self.client.stop_server(server_name=server_name)

    def upload(self, server_name: str = None, file_path: str = "", remote_path=""):
        """
        Uploads a file to a remote server.

        Args:
            server_name (str): The name of the server to which the file will be uploaded. If None, uses the server name set in the instance.
            file_path (str): The local path to the file.
            remote_path (str): The remote path where the file will be stored.

        Returns:
            The response from the upload request.
        """
        if not self.api_key:
            raise ValueError("Requires user authentication with an API Key")
        # Use set self.server_name if not provided
        if server_name is None:
            server_name = self.server_name
        return self.client.put_contents(
            server_name=server_name, path=remote_path, file_path=file_path
        )

    def run_code(
        self,
        code: str = "print('Hello world!')",
        server_name: str = None,
        clear_other_kernels: bool = True,
        return_full: bool = False,
    ):
        """
        Executes the given code on a remote server.

        Args:
            code (str): The code to execute. If a file path is provided, the code in the file is executed.
            server_name (str): The name of the server where the code will be executed. If None, uses the server name set in the instance.
            clear_other_kernels (bool): Whether to shut down other kernels on the server before executing the code.
            return_full (bool): Whether to return the full response from the server.

        Returns:
            The output from the code execution.
        """
        # Use set self.server_name if not provided
        if server_name is None:
            server_name = self.server_name
        if clear_other_kernels:
            self.client.shutdown_all_kernels(server_name=server_name)
        if os.path.isfile(code):
            if os.path.splitext(code)[1] != ".py":
                warnings.warn(
                    "This doesn't seem to be a python file, but will try to run it anyway."
                )
            with open(code, "r") as f:
                code = f.read()
        return self.client.send_channel_execute(
            server_name=server_name, messages=[code], return_full=return_full
        )

    def get_service(self, server_name: str = None):
        """
        Retrieves the service URL for a running service or app on the server.

        Args:
            server_name (str): The name of the server. If None, uses the server name set in the instance.

        Returns:
            The URL to access the running service or app on the server.
        """
        if not self.api_key:
            raise ValueError("Requires user authentication with an API Key")
        if server_name is None:
            server_name = self.server_name
        return f"If you have running service or app in your server, check it out here -> {self.client.get_service(server_name=server_name)}"

    # Profilers
    def get_profiler_stats(self):
        """
        Retrieves statistics from the PyTorch profiler.

        Returns:
            A table containing key averages of profiler statistics, particularly focusing on CUDA time.
        """
        stat_table = self.pt_profiler.key_averages().table(
            sort_by="cuda_time_total", row_limit=10
        )
        return stat_table

    def get_averages(
        self,
        sort_by="cuda_time_total",
        header=None,
        row_limit=100,
        max_src_column_width=75,
        max_name_column_width=55,
        max_shapes_column_width=80,
        top_level_events_only=False,
    ):
        """
        Retrieves a DataFrame of average statistics from the PyTorch profiler powered by Kineto.

        Args:
            sort_by (str): The attribute to sort the data by. Defaults to 'cuda_time_total'.
            header (str, optional): Header for the DataFrame. Defaults to None.
            row_limit (int): The maximum number of rows to return. Defaults to 100.
            max_src_column_width (int): Maximum width for the source column. Defaults to 75.
            max_name_column_width (int): Maximum width for the name column. Defaults to 55.
            max_shapes_column_width (int): Maximum width for the shapes column. Defaults to 80.
            top_level_events_only (bool): Whether to include only top-level events. Defaults to False.

        Returns:
            pandas.DataFrame: A DataFrame containing the averaged profiler statistics.
        """
        df_averages, self.dict_averages, str_averages = self.pt_profiler.get_averages(
            sort_by="cuda_time_total",
            header=None,
            row_limit=100,
            max_src_column_width=75,
            max_name_column_width=55,
            max_shapes_column_width=80,
            top_level_events_only=False,
        )
        return df_averages

    def run_profiler_analysis(self, trace_path: str = None, visualize: bool = False):
        """
        Runs an analysis of the profiler data and optionally visualizes the results.

        Args:
            trace_path (str, optional): The path to the trace data. If None, uses the latest trace. Defaults to None.
            visualize (bool): Whether to visualize the analysis results. Defaults to False.

        Returns:
            A tuple of DataFrames containing various analysis results, such as idle time, temporal breakdown, and GPU kernel breakdown.
        """
        if trace_path:
            pt_trace_dirs = [trace_path]
        else:
            pt_trace_dirs = self.get_pt_traces()
        if pt_trace_dirs:
            try:
                trace_dir = os.path.join(self.pt_profiler.dir_path, pt_trace_dirs[-1])
                self.analyzer = TraceAnalysis(trace_dir=trace_dir)
                idle_time_df = self.analyzer.get_idle_time_breakdown(
                    show_idle_interval_stats=True, visualize=visualize
                )
                time_spent_df = self.analyzer.get_temporal_breakdown(
                    visualize=visualize
                )
                (
                    kernel_type_metrics_df,
                    kernel_metrics_df,
                ) = self.analyzer.get_gpu_kernel_breakdown()
                self.dict_idle_time = idle_time_df[0].to_dict()
                self.dict_time_spent = time_spent_df.to_dict()
                self.dict_type_metrics = kernel_type_metrics_df.to_dict()
                self.dict_kernel_metrics = kernel_metrics_df.to_dict()
                return (
                    idle_time_df,
                    time_spent_df,
                    kernel_type_metrics_df,
                    kernel_metrics_df,
                )
            except:
                warnings.warn(
                    "Error running profiler analysis, may not have GPU trace data so will continue without it."
                )
                self.dict_idle_time = {}
                self.dict_time_spent = {}
                self.dict_type_metrics = {}
                self.dict_kernel_metrics = {}
                return

    def save(self, return_results: bool = False):
        """
        Saves the profiler data and analysis results to a specified directory.

        Args:
            return_results (bool): Whether to return the saved data as a dictionary. Defaults to False.

        Returns:
            dict (optional): A dictionary containing the saved data if return_results is True.
        """
        alphai_dict = {}
        if self.dict_idle_time is None:
            warnings.warn(
                "Make sure to run_profiler_analysis() before saving to your analytics."
            )
            self.run_profiler_analysis()
        self.get_averages()
        alphai_dict["metadata"] = self.analyzer.t.meta_data
        alphai_dict["idle_time"] = self.dict_idle_time
        alphai_dict["time_spent"] = self.dict_time_spent
        alphai_dict["type_metrics"] = self.dict_type_metrics
        alphai_dict["kernel_metrics"] = self.dict_kernel_metrics
        alphai_dict["key_averages"] = self.dict_averages
        with open(
            os.path.join(self.pt_profiler.profiler_path, "profiling.alphai"), "w"
        ) as f:
            json.dump(alphai_dict, f, indent=4)
        if return_results:
            return alphai_dict

    def load_view(self, dir_name: str = None):
        """
        Loads a view of the profiler data onto your remote server.

        Args:
            dir_name (str, optional): The directory name to load the view from. If None, generates a timestamp-based directory name. Defaults to None.

        Returns:
            str: A URL to the GPU usage statistics dashboard.
        """
        if not self.api_key:
            raise ValueError("Requires user authentication with an API Key")
        formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if not dir_name:
            view_path = f"{formatted_datetime}.alphai"
        else:
            view_path = dir_name
        self.client.post_contents(path="", ext=".alphai", type="directory")
        self.client.patch_contents(path="Untitled Folder.alphai", new_path=view_path)
        self.client.put_contents(
            path=view_path,
            file_path=f"{self.pt_profiler.profiler_path}/profiling.alphai",
        )
        return f"Check out your GPU usage statistics at -> https://dashboard.amdatascience.com/agent-alph"

    def get_pt_traces(self):
        """
        Retrieves a list of PyTorch trace directories sorted by date.

        Returns:
            List[str]: A list of directory names containing PyTorch traces.
        """
        # List all items in the directory
        directory_path = self.output_path
        if not os.path.isdir(directory_path):
            return []
        all_items = os.listdir(directory_path)

        # Filter out items that are directories and follow the naming pattern
        date_directories = []
        for item in all_items:
            if os.path.isdir(os.path.join(directory_path, item)) and item.startswith(
                "pt_trace_"
            ):
                # Extract the date and time part from the folder name
                datetime_part = item.split("pt_trace_")[1]

                # Parse the date and time into a datetime object
                try:
                    folder_date = datetime.datetime.strptime(
                        datetime_part, "%Y-%m-%d_%H-%M-%S"
                    )
                    date_directories.append((item, folder_date))
                except ValueError:
                    # Handle cases where the date format is incorrect or different
                    print(f"Skipping {item} due to unexpected date format.")

        # Sort the directories by the parsed datetime
        date_directories.sort(key=lambda x: x[1])

        # Return only the directory names, in sorted order
        return [name for name, date in date_directories]

    def get_jax_traces(self):
        """
        Retrieves a list of JAX trace directories sorted by date.

        Returns:
            List[str]: A list of directory names containing JAX traces.
        """
        # List all items in the directory
        directory_path = self.output_path
        if not os.path.isdir(directory_path):
            return []
        all_items = os.listdir(directory_path)

        # Filter out items that are directories and follow the naming pattern
        date_directories = []
        for item in all_items:
            if os.path.isdir(os.path.join(directory_path, item)) and item.startswith(
                "jax_trace_"
            ):
                # Extract the date and time part from the folder name
                datetime_part = item.split("jax_trace_")[1]

                # Parse the date and time into a datetime object
                try:
                    folder_date = datetime.datetime.strptime(
                        datetime_part, "%Y-%m-%d_%H-%M-%S"
                    )
                    date_directories.append((item, folder_date))
                except ValueError:
                    # Handle cases where the date format is incorrect or different
                    print(f"Skipping {item} due to unexpected date format.")

        # Sort the directories by the parsed datetime
        date_directories.sort(key=lambda x: x[1])

        # Return only the directory names, in sorted order
        return [name for name, date in date_directories]

    # Benchmarker
    def start_timer(self):
        """
        Starts the benchmarking timer.
        """
        self.benchmarker.start()

    def stop_timer(self, print_results: bool = True):
        """
        Stops the timer and optionally prints the results.

        Args:
            print_results (bool): Whether to print the results. Defaults to True.

        Returns:
            The results of the benchmark.
        """
        return self.benchmarker.stop()

    def benchmark(
        self,
        function: Callable = None,
        *args,
        num_iter: int = 100,
        print_results: bool = True,
        **kwargs,
    ):
        """
        Benchmarks a function by running it a specified number of times.

        Args:
            function (Callable): The function to benchmark.
            *args: The arguments to pass to the function.
            num_iter (int): The number of times to run the function. Defaults to 100.
            print_results (bool): Whether to print the results. Defaults to True.
            **kwargs: The keyword arguments to pass to the function.

        Returns:
            The results of the benchmark.
        """
        return self.benchmarker.benchmark(
            function, *args, num_iter=num_iter, print_results=print_results, **kwargs
        )

    # Hugging Face utility

    def estimate_memory_requirement(
        self,
        model_name: str = "stabilityai/stablelm-zephyr-3b",
    ):
        """
        Estimates the memory requirement for a given model.

        Args:
            model_name (str): The name of the model. Defaults to "stabilityai/stablelm-zephyr-3b".

        Returns:
            A dictionary with the model name and the estimated memory requirement in MB and GB.
        """
        try:
            param_value = extract_param_value(model_name)
            megabyte_value = param_value * 2 * 1000
            gigabyte_value = param_value * 2
            print(
                f"Estimated memory requirement assuming float16 dtype for {model_name}: {megabyte_value:.2f} MB or {gigabyte_value:.2f} GB"
            )
            return {
                "model_name_or_path": model_name,
                "estimate_memory_requirement_mb_float16": f"{megabyte_value:.2f} MB",
                "estimate_memory_requirement_gb_float16": f"{gigabyte_value:.2f} GB",
            }
        except:
            warnings.warn(
                "Error parsing model name or path, can't estimate memory requirement."
            )
            return

    def memory_requirement(
        self,
        model_name_or_path: str = "stabilityai/stablelm-zephyr-3b",
        device: str = "cuda",
        trust_remote_code=True,
        torch_dtype="auto",
    ):
        """
        Estimates and prints the memory requirement for a specified model.

        Args:
            model_name_or_path (str): The name or path of the model to be loaded. Defaults to 'stabilityai/stablelm-zephyr-3b'.
            device (str): The device to load the model on ('cuda' or 'cpu'). Defaults to 'cuda'.
            trust_remote_code (bool): Whether to trust remote code when loading the model. Defaults to True.
            torch_dtype (str): The data type for the model parameters. Defaults to 'auto'.

        Returns:
            dict: A dictionary containing the memory requirement in MB and GB.
        """
        if not is_package_installed("torch"):
            warnings.warn(f"You need to install 'torch' to run memory_requirement")
            return
        if not self.model_name_or_path or self.model_name_or_path != model_name_or_path:
            self.model_name_or_path = model_name_or_path
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch_dtype,
                ).to(device)
            except:
                warnings.warn(
                    "Loading model to CPU instead of GPU since GPU is not available."
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch_dtype,
                ).to("cpu")
        try:
            param_value = self.model.num_parameters()
        except:
            param_value = sum(p.numel() for p in self.model.parameters())

        megabyte_value = param_value * 2 / 1000000
        gigabyte_value = param_value * 2 / 1000000000
        print(
            f"Memory requirement assuming float16 dtype for {model_name_or_path}: {megabyte_value:.2f} MB or {gigabyte_value:.2f} GB"
        )
        return {
            "model_name_or_path": model_name_or_path,
            "memory_requirement_mb_float16": f"{megabyte_value:.2f} MB",
            "memory_requirement_gb_float16": f"{gigabyte_value:.2f} GB",
        }

    def generate(
        self,
        text: str = "",
        prompt: List[dict] = None,
        model_name_or_path: str = "stabilityai/stablelm-zephyr-3b",
        trust_remote_code=True,
        torch_dtype="auto",
        stream: bool = True,
        device: str = "cuda",
        **kwargs,
    ):
        """
        Generates text using the specified model based on the given prompt or text.

        Args:
            text (str): The text to be used as a prompt. Defaults to an empty string.
            prompt (List[dict]): A list of dictionaries defining the prompt structure. Defaults to None.
            model_name_or_path (str): The name or path of the model to be used. Defaults to 'stabilityai/stablelm-zephyr-3b'.
            trust_remote_code (bool): Whether to trust remote code when loading the model. Defaults to True.
            torch_dtype (str): The data type for the model parameters. Defaults to 'auto'.
            stream (bool): Whether to use streaming for text generation. Defaults to True.
            device (str): The device to run the model on. Defaults to 'cuda'.

        Returns:
            str: The generated text.
        """
        if not is_package_installed("torch"):
            warnings.warn(f"You need to install 'torch' to run generate")
            return
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        streamer = TextStreamer(tokenizer) if stream else None
        if not self.model_name_or_path or self.model_name_or_path != model_name_or_path:
            self.model_name_or_path = model_name_or_path
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch_dtype,
                ).to(device)
            except:
                warnings.warn(
                    "Loading model to CPU instead of GPU since GPU is not available."
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch_dtype,
                ).to("cpu")

        if not prompt:
            prompt = [{"role": "user", "content": text}]
        inputs = tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, return_tensors="pt"
        )

        tokens = self.model.generate(
            inputs.to(self.model.device),
            max_new_tokens=1024,
            temperature=0.8,
            do_sample=True,
            streamer=streamer,
            **kwargs,
        )

        return tokenizer.decode(tokens[0])

    def clear_cuda_memory(self):
        """
        Clears the CUDA memory cache to free up GPU memory.

        Raises:
            Warning: If PyTorch is not installed.
        """
        if not is_package_installed("torch"):
            warnings.warn(f"You need to install 'torch' to run clear_cuda_memory")
            return
        gc.collect()
        torch.cuda.empty_cache()
