import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Callable


def profile_pipeline(
    generate_fn: Callable,
    train_step_fn: Callable,
    sort_by: str = "cuda_time_total",
    row_limit: int = 20,
):
    """
    Profile one iteration of generate + train_step.

    Args:
        generate_fn:   callable that runs self-play generation
        train_step_fn: callable that runs one training step
        sort_by:       profiler sort key (e.g. 'cuda_time_total', 'cpu_time_total')
        row_limit:     number of rows to print
    """
    activities = [ProfilerActivity.CPU]
    if torch.backends.mps.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("generate"):
            generate_fn()
        with record_function("train_step"):
            train_step_fn()

    print(prof.key_averages().table(sort_by=sort_by, row_limit=row_limit))