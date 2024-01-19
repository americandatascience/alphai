# src/alphai/profilers/configs_base.py
from dataclasses import dataclass


@dataclass
class BaseProfilerConfigs:
    dir_path: str = "./.alphai"
