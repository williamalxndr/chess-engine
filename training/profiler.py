import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Callable
import psutil
import os
import threading
import time


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

def profile_cpu(generate_fn, interval=0.1):
    samples = []
    stop_event = threading.Event()

    def monitor():
        while not stop_event.is_set():
            per_core = psutil.cpu_percent(percpu=True)
            samples.append(per_core)
            time.sleep(interval)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    t0 = time.time()
    generate_fn()
    elapsed = time.time() - t0

    stop_event.set()
    thread.join()

    # Summary
    num_cores = len(samples[0]) if samples else 0
    avg_per_core = [sum(s[i] for s in samples) / len(samples) for i in range(num_cores)]

    print(f"\nCPU Monitor ({elapsed:.2f}s, {len(samples)} samples)")
    print(f"{'Core':<8} {'Avg %':<10} {'Bar'}")
    print("-" * 40)
    for i, avg in enumerate(avg_per_core):
        bar = "█" * int(avg / 5)
        print(f"Core {i:<3} {avg:>6.1f}%   {bar}")
    print(f"\nAvg total: {sum(avg_per_core)/num_cores:.1f}% across {num_cores} cores")
    print(f"Active cores (>10%): {sum(1 for a in avg_per_core if a > 10)}/{num_cores}")