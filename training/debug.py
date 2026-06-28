# this file is just for debugging how much speed up data parallel gets if compared using single GPU

import time
import torch

from core import factory
from core.config import load_config
from training.pipeline import Pipeline
from profiler import timing as prof

cfg = load_config(["--config", "configs/kaggle-test.yaml"])

import time
from core import factory
from core.config import load_config
from training.pipeline import Pipeline

cfg = load_config(["--config", "configs/kaggle-test.yaml"])

def run(dp: bool) -> float:
    network = factory.build_network(cfg.game, version=cfg.version, dp=dp)

    pipeline = Pipeline(
        game=cfg.game,
        version=cfg.version,
        network=network,
        save_file_name=cfg.file_name,
        iterations=99999,
        steps_per_iter=cfg.training.steps_per_iter,
        train_batch_size=cfg.training.train_batch_size,
        mcts_batch_size=cfg.mcts.mcts_batch_size,
        num_rollout=cfg.mcts.num_rollout,
        num_selfplay=cfg.mcts.num_selfplay,
        replay_buffer_max_size=cfg.training.replay_buffer_max_size,
        verbose=cfg.logging.verbose,
        kaggle=cfg.logging.kaggle,
    )

    start = time.time()
    pipeline.train(duration_hours=cfg.training.duration, log_interval=cfg.logging.log_interval)
    return time.time() - start


if __name__ == "__main__": 
    num_gpu = torch.cuda.device_count() 
    print("DEBUG]: DataParallel vs Single GPU \n")

    # Running single GPU Training
    prof.reset()
    print("Running Single GPU...")
    t_single = run(dp=False)
    print(prof.summary())


    # Running multiple GPU Training (Data Parallel only)
    prof.reset()
    print(f"Running on {num_gpu} GPU...")
    t_multi = run(dp=True)
    print(prof.summary())


    # Comparison
    print(f"Single GPU : {t_single:.1f}s\n")
    print(f"Multi  GPU : {t_multi:.1f}s\n")

    print(f"Speedup    : {t_single / t_multi:.2f}x")