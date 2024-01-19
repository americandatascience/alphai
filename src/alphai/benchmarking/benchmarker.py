from typing import Callable

import timeit
from timeit import default_timer


class Benchmarker:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0

    def start(self):
        self.start_time = default_timer()

    def stop(self, print_results: bool = True):
        self.end_time = default_timer()
        delta = self.end_time - self.start_time
        print(f"Measured: {delta * 1e6:.1f} us")
        return delta

    def benchmark(
        self,
        function: Callable = None,
        *args,
        num_iter: int = 1,
        print_results: bool = True,
        **kwargs,
    ):
        results = {}
        total_time = timeit.Timer(lambda: function(*args, **kwargs)).timeit(num_iter)
        avg_run = total_time / num_iter
        results["total_time_seconds"] = total_time
        results["avg_run_seconds"] = avg_run
        if print_results:
            print(f"Measured total run: {total_time * 1e6:>5.1f} us")
            print(f"Measured averaged run: {avg_run * 1e6:>5.1f} us")
        return results
