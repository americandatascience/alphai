# src/alphai/profilers/pytorch.py
import os
import datetime
import json
from typing import Optional, Callable, Iterable, Any
from dataclasses import dataclass, field

import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    ProfilerAction,
    _ExperimentalConfig,
)

from alphai.profilers.configs_base import BaseProfilerConfigs
from alphai.profilers.pytorch_utils import _build_dataframe


@dataclass
class PyTorchProfilerConfigs(BaseProfilerConfigs):
    dir_path: str = "./.alphai"
    activities: Optional[Iterable[ProfilerActivity]] = field(
        default_factory=lambda: [
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ]
    )
    schedule: Optional[Callable[[int], ProfilerAction]] = None
    on_trace_ready: Optional[
        Callable[..., Any]
    ] = torch.profiler.tensorboard_trace_handler(dir_path)
    record_shapes: bool = False  # True
    profile_memory: bool = True  # True
    with_stack: bool = False
    with_flops: bool = True
    with_modules: bool = False
    experimental_config: Optional[_ExperimentalConfig] = None

    def as_dict(self):
        return {key: vars(self)[key] for key in vars(self) if key != "dir_path"}


class PyTorchProfiler(profile):
    def __init__(self, configs: PyTorchProfilerConfigs, **kwargs):
        super().__init__(**configs.as_dict())
        self.dir_path = configs.dir_path

    def start(self, dir_name: str = None):
        super().start()
        if not self._get_distributed_info():
            self.add_metadata_json("distributedInfo", json.dumps({"rank": 0}))
        formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if not dir_name:
            dir_name = f"pt_trace_{formatted_datetime}"
        self.profiler_path = os.path.join(self.dir_path, dir_name)
        self.on_trace_ready = torch.profiler.tensorboard_trace_handler(
            self.profiler_path
        )

    def get_averages(
        self,
        sort_by="cuda_time_total",
        header=None,
        row_limit=100,
        max_src_column_width=75,
        max_name_column_width=55,
        max_shapes_column_width=80,
        top_level_events_only=False,
        **kwargs,
    ):
        return _build_dataframe(
            self.events().key_averages(),
            sort_by=sort_by,
            header=header,
            row_limit=row_limit,
            max_src_column_width=max_src_column_width,
            max_name_column_width=max_name_column_width,
            max_shapes_column_width=max_shapes_column_width,
            with_flops=self.events()._with_flops,
            profile_memory=self.events()._profile_memory,
            top_level_events_only=top_level_events_only,
        )
