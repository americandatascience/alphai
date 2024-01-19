# src/alphai/profilers/jax.py
import os
from dataclasses import dataclass, field
import datetime

from alphai.profilers.configs_base import BaseProfilerConfigs
from jax.profiler import start_trace, stop_trace, trace


@dataclass
class JaxProfilerConfigs(BaseProfilerConfigs):
    dir_path: str = "./.alphai"
    create_perfetto_link: bool = False
    create_perfetto_trace: bool = False

    def as_dict(self):
        return {key: vars(self)[key] for key in vars(self) if key != "dir_path"}


class JaxProfiler:
    def __init__(self, configs: JaxProfilerConfigs, **kwargs):
        self.configs = configs
        self.dir_path = configs.dir_path

    def start(self, dir_name: str = None):
        formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if not dir_name:
            dir_name = f"jax_trace_{formatted_datetime}"
        profiler_path = os.path.join(self.dir_path, dir_name)
        start_trace(
            log_dir=profiler_path,
            create_perfetto_link=self.configs.create_perfetto_link,
            create_perfetto_trace=self.configs.create_perfetto_trace,
        )

    def stop(self):
        stop_trace()
