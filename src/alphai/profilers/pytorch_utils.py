# src/alphai/profilers/pytorch_utils.py
import math

import pandas as pd
from torch.autograd import (
    _disable_profiler,
    _enable_profiler,
    _kineto_step,
    _prepare_profiler,
    _ProfilerResult,
    _supported_activities,
    DeviceType,
    kineto_available,
    ProfilerActivity,
    ProfilerConfig,
    ProfilerState,
)
from torch.autograd.profiler_util import (
    _filter_name,
    _filter_stack_entry,
    _rewrite_name,
    EventList,
    FunctionEvent,
    MEMORY_EVENT_NAME,
    MemRecordsAcc,
    OUT_OF_MEMORY_EVENT_NAME,
)


def _format_time(time_us):
    """Define how to format time in FunctionEvent."""
    US_IN_SECOND = 1000.0 * 1000.0
    US_IN_MS = 1000.0
    if time_us >= US_IN_SECOND:
        return f"{time_us / US_IN_SECOND:.3f}s"
    if time_us >= US_IN_MS:
        return f"{time_us / US_IN_MS:.3f}ms"
    return f"{time_us:.3f}us"


def _format_time_share(time_us, total_time_us):
    """Define how to format time in FunctionEvent."""
    if total_time_us == 0:
        assert time_us == 0, f"Expected time_us == 0 but got {time_us}"
        return "NaN"
    return f"{time_us * 100.0 / total_time_us:.2f}%"


def _format_memory(nbytes):
    """Return a formatted memory size string."""
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    if abs(nbytes) >= GB:
        return f"{nbytes * 1.0 / GB:.2f} Gb"
    elif abs(nbytes) >= MB:
        return f"{nbytes * 1.0 / MB:.2f} Mb"
    elif abs(nbytes) >= KB:
        return f"{nbytes * 1.0 / KB:.2f} Kb"
    else:
        return str(nbytes) + " b"


def _attr_formatter(name):
    return property(lambda self: _format_time(getattr(self, name)))


def _build_dataframe(
    events,
    sort_by="cuda_time_total",
    header=None,
    row_limit=10,
    max_src_column_width=75,
    max_name_column_width=55,
    max_shapes_column_width=80,
    with_flops=False,
    profile_memory=False,
    top_level_events_only=False,
):
    """Print a summary of events (which can be a list of FunctionEvent or FunctionEventAvg)."""
    if len(events) == 0:
        return ""

    has_cuda_time = any(event.self_cuda_time_total > 0 for event in events)
    has_cuda_mem = any(event.self_cuda_memory_usage > 0 for event in events)
    has_privateuse1_time = any(
        event.self_privateuse1_time_total > 0 for event in events
    )
    has_privateuse1_mem = any(
        event.self_privateuse1_memory_usage > 0 for event in events
    )
    use_device = events[0].use_device
    if not use_device and (has_privateuse1_mem or has_privateuse1_time):
        raise RuntimeError(
            "use_device is None, but there is private device performance data."
        )

    has_input_shapes = any(
        (event.input_shapes is not None and len(event.input_shapes) > 0)
        for event in events
    )

    if sort_by is not None:
        events = EventList(
            sorted(events, key=lambda evt: getattr(evt, sort_by), reverse=True),
            use_cuda=has_cuda_time,
            use_device=use_device,
            profile_memory=profile_memory,
            with_flops=with_flops,
        )

    name_column_width = max([len(evt.key) for evt in events]) + 4
    if max_name_column_width is not None:
        name_column_width = min(name_column_width, max_name_column_width)

    shapes_column_width = max([len(str(evt.input_shapes)) for evt in events]) + 4
    if max_shapes_column_width is not None:
        shapes_column_width = min(shapes_column_width, max_shapes_column_width)

    DEFAULT_COLUMN_WIDTH = 12
    flops_column_width = DEFAULT_COLUMN_WIDTH

    src_column_width = None
    stacks = []
    for evt in events:
        if evt.stack is not None and len(evt.stack) > 0:
            stacks.append(evt.stack)
    has_stack = len(stacks) > 0
    if has_stack:
        src_column_width = (
            max([max([len(entry) for entry in stack]) for stack in stacks]) + 4
        )
        if max_src_column_width is not None:
            src_column_width = min(src_column_width, max_src_column_width)

    headers = [
        "Name",
        "Self CPU %",
        "Self CPU",
        "CPU total %",
        "CPU total",
        "CPU time avg",
    ]
    if has_cuda_time:
        headers.extend(
            [
                "Self CUDA",
                "Self CUDA %",
                "CUDA total",
                "CUDA time avg",
            ]
        )
    if has_privateuse1_time:
        privateuse1 = use_device.upper()
        headers.extend(
            [
                f"Self {privateuse1}",
                f"Self {privateuse1} %",
                f"{privateuse1} total",
                f"{privateuse1} time avg",
            ]
        )
    if profile_memory:
        headers.extend(
            [
                "CPU Mem",
                "Self CPU Mem",
            ]
        )
        if has_cuda_mem:
            headers.extend(
                [
                    "CUDA Mem",
                    "Self CUDA Mem",
                ]
            )
        if has_privateuse1_mem:
            privateuse1 = use_device.upper()
            headers.extend(
                [
                    f"{privateuse1} Mem",
                    f"Self {privateuse1} Mem",
                ]
            )
    headers.append("# of Calls")
    # Only append Node ID if any event has a valid (>= 0) Node ID
    append_node_id = any(evt.node_id != -1 for evt in events)
    if append_node_id:
        headers.append("Node ID")

    # Have to use a list because nonlocal is Py3 only...
    SPACING_SIZE = 2
    row_format_lst = [""]
    header_sep_lst = [""]
    line_length_lst = [-SPACING_SIZE]
    MAX_STACK_ENTRY = 5

    def add_column(padding, text_dir=">"):
        row_format_lst[0] += (
            "{: " + text_dir + str(padding) + "}" + (" " * SPACING_SIZE)
        )
        header_sep_lst[0] += "-" * padding + (" " * SPACING_SIZE)
        line_length_lst[0] += padding + SPACING_SIZE

    def auto_scale_flops(flops):
        flop_headers = [
            "FLOPs",
            "KFLOPs",
            "MFLOPs",
            "GFLOPs",
            "TFLOPs",
            "PFLOPs",
        ]
        assert flops > 0
        log_flops = max(0, min(math.log10(flops) / 3, float(len(flop_headers) - 1)))
        assert log_flops >= 0 and log_flops < len(flop_headers)
        return (pow(10, (math.floor(log_flops) * -3.0)), flop_headers[int(log_flops)])

    add_column(name_column_width)
    for _ in headers[1:]:
        add_column(DEFAULT_COLUMN_WIDTH)

    if has_input_shapes:
        headers.append("Input Shapes")
        add_column(shapes_column_width)

    if has_stack:
        headers.append("Source Location")
        add_column(src_column_width, text_dir="<")

    if with_flops:
        # Auto-scaling of flops header
        raw_flops = []
        for evt in events:
            if evt.flops > 0:
                raw_flops.append(evt.flops)
        if len(raw_flops) != 0:
            (flops_scale, flops_header) = auto_scale_flops(min(raw_flops))
            headers.append(f"Total {flops_header}")
            add_column(flops_column_width)
        else:
            with_flops = False  # can't find any valid flops

    row_format = row_format_lst[0]
    header_sep = header_sep_lst[0]
    line_length = line_length_lst[0]
    add_column = None  # type: ignore[assignment]

    # Have to use a list because nonlocal is Py3 only...
    result = []

    def append(s):
        result.append(s)
        result.append("\n")  # Yes, newline after the end as well

    sum_self_cpu_time_total = sum([event.self_cpu_time_total for event in events])
    sum_self_cuda_time_total = 0
    sum_self_privateuse1_time_total = 0
    for evt in events:
        if evt.device_type == DeviceType.CPU:
            # in legacy profiler, kernel info is stored in cpu events
            if evt.is_legacy:
                if not use_device:
                    sum_self_cuda_time_total += evt.self_cuda_time_total
                else:
                    sum_self_privateuse1_time_total += evt.self_privateuse1_time_total
        elif evt.device_type == DeviceType.CUDA:
            # in kineto profiler, there're events with the correct device type (e.g. CUDA)
            sum_self_cuda_time_total += evt.self_cuda_time_total
        elif evt.device_type == DeviceType.PrivateUse1:
            sum_self_privateuse1_time_total += evt.self_privateuse1_time_total

    # Actual printing
    if header is not None:
        append("=" * line_length)
        append(header)
    if top_level_events_only:
        append("=" * line_length)
        append("This report only display top-level ops statistics")
    append(header_sep)
    append(row_format.format(*headers))

    append(header_sep)

    def trim_path(path, src_column_width):
        if len(path) > src_column_width:
            offset = len(path) - src_column_width
            path = path[offset:]
            if len(path) > 3:
                path = "..." + path[3:]
        return path

    df_data = []
    event_limit = 0
    for evt in events:
        if event_limit == row_limit:
            break
        if top_level_events_only and evt.cpu_parent is not None:
            continue
        else:
            event_limit += 1
        name = evt.key
        if max_name_column_width is not None and len(name) >= max_name_column_width - 3:
            name = name[: (max_name_column_width - 3)] + "..."
        row_values = [
            name,
            # Self CPU total %, 0 for async events.
            _format_time_share(evt.self_cpu_time_total, sum_self_cpu_time_total),
            evt.self_cpu_time_total_str,  # Self CPU total
            # CPU total %, 0 for async events.
            _format_time_share(evt.cpu_time_total, sum_self_cpu_time_total)
            if not evt.is_async
            else 0,
            evt.cpu_time_total_str,  # CPU total
            evt.cpu_time_str,  # CPU time avg
        ]
        if has_cuda_time:
            row_values.extend(
                [
                    evt.self_cuda_time_total_str,
                    # CUDA time total %
                    _format_time_share(
                        evt.self_cuda_time_total, sum_self_cuda_time_total
                    ),
                    evt.cuda_time_total_str,
                    evt.cuda_time_str,  # Cuda time avg
                ]
            )
        if has_privateuse1_time:
            row_values.extend(
                [
                    evt.self_privateuse1_time_total_str,
                    # PrivateUse1 time total %
                    _format_time_share(
                        evt.self_privateuse1_time_total, sum_self_privateuse1_time_total
                    ),
                    evt.privateuse1_time_total_str,
                    evt.privateuse1_time_str,  # PrivateUse1 time avg
                ]
            )
        if profile_memory:
            row_values.extend(
                [
                    # CPU Mem Total
                    _format_memory(evt.cpu_memory_usage),
                    # Self CPU Mem Total
                    _format_memory(evt.self_cpu_memory_usage),
                ]
            )
            if has_cuda_mem:
                row_values.extend(
                    [
                        # CUDA Mem Total
                        _format_memory(evt.cuda_memory_usage),
                        # Self CUDA Mem Total
                        _format_memory(evt.self_cuda_memory_usage),
                    ]
                )
            if has_privateuse1_mem:
                row_values.extend(
                    [
                        # PrivateUse1 Mem Total
                        _format_memory(evt.privateuse1_memory_usage),
                        # Self PrivateUse1 Mem Total
                        _format_memory(evt.self_privateuse1_memory_usage),
                    ]
                )
        row_values.append(
            evt.count,  # Number of calls
        )

        if append_node_id:
            row_values.append(evt.node_id)
        if has_input_shapes:
            row_values.append(str(evt.input_shapes)[:shapes_column_width])
        if with_flops:
            if evt.flops <= 0:
                row_values.append("--")
            else:
                row_values.append(f"{evt.flops * flops_scale:8.3f}")
        if has_stack:
            src_field = ""
            if len(evt.stack) > 0:
                src_field = trim_path(evt.stack[0], src_column_width)
            row_values.append(src_field)
        append(row_format.format(*row_values))

        if has_stack:
            empty_headers = [""] * (len(headers) - 1)
            for entry in evt.stack[1:MAX_STACK_ENTRY]:
                append(
                    row_format.format(
                        *(empty_headers + [trim_path(entry, src_column_width)])
                    )
                )
            empty_headers.append("")
            append(row_format.format(*empty_headers))
        df_data.append(row_values)

    df = pd.DataFrame(df_data, columns=headers)
    dict_results = {}
    dict_results["df_data"] = df.to_dict()
    append(header_sep)
    self_cpu_time_total = (
        f"Self CPU time total: {_format_time(sum_self_cpu_time_total)}"
    )
    append(self_cpu_time_total)
    dict_results["self_cpu_time_total"] = self_cpu_time_total
    if has_cuda_time:
        self_cuda_time_total = (
            f"Self CUDA time total: {_format_time(sum_self_cuda_time_total)}"
        )
        append(self_cuda_time_total)
        dict_results["self_cuda_time_total"] = self_cuda_time_total
    if has_privateuse1_time:
        self_privateuse1_time_total = f"Self {use_device.upper()} time total: {_format_time(sum_self_privateuse1_time_total)}"
        append(self_privateuse1_time_total)
        dict_results["self_privateuse1_time_total"] = self_privateuse1_time_total
    return df, dict_results, "".join(result)
