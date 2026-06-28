import time

batch_selection_time = 0.0
inference_time = 0.0
node_backprop_time = 0.0
network_optimize_time = 0.0

def add(batch_selection=0.0, inference=0.0, node_backprop=0.0, network_optimize=0.0):
    global batch_selection_time, inference_time, node_backprop_time, network_optimize_time
    batch_selection_time  += batch_selection
    inference_time        += inference
    node_backprop_time    += node_backprop
    network_optimize_time += network_optimize


def reset():
    global batch_selection_time, inference_time, node_backprop_time, network_optimize_time
    batch_selection_time = 0.0
    inference_time = 0.0
    node_backprop_time = 0.0
    network_optimize_time = 0.0

def summary():
    total = batch_selection_time + inference_time + node_backprop_time + network_optimize_time
    cpu_time = batch_selection_time + node_backprop_time
    gpu_time = inference_time + network_optimize_time
    return (
        f"  [SELFPLAY] Batch selection time    (CPU): {batch_selection_time:.2f}s\n"
        f"  [SELFPLAY] Inference time          (GPU): {inference_time:.2f}s\n"
        f"  [SELFPLAY] Node backprop time      (CPU): {node_backprop_time:.2f}s\n"
        f"  [OPTIMIZER] Network optimize time  (GPU): {network_optimize_time:.2f}s"
        f"  CPU Time: {cpu_time:.2f}"
        f"  GPU Time: {gpu_time:.2f}s"
        f"  Total: {total}"
    )