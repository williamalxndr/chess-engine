# this file is just for debugging how much speed up data parallel gets if compared using single GPU

import time
import torch

from core import factory
from core.config import load_config
from training.pipeline import Pipeline
from profiler import timing as prof

cfg = load_config()

def run(dp: bool) -> float:
    prof.reset()

    network = factory.build_network(cfg.game, version=cfg.version, dp=dp)

    pipeline = Pipeline(
        game=cfg.game,
        version=cfg.version,
        network=network,
        save_file_name=cfg.file_name,
        iterations=cfg.iterations or 99999,
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
    elapsed = time.time() - start

    print(prof.summary())
    return elapsed


if __name__ == "__main__":
    num_gpu = torch.cuda.device_count()
    print("Running Single GPU...")
    t_single = run(dp=False)
    print(f"Single GPU : {t_single:.1f}s\n")

    print(f"Running on {num_gpu} GPU...")
    t_multi = run(dp=True)
    print(f"Multi  GPU : {t_multi:.1f}s\n")

    print(f"Speedup    : {t_single / t_multi:.2f}x")