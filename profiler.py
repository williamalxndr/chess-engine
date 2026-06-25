import cProfile

from training.pipeline import Pipeline
from core.network import NetworkFactory


net = NetworkFactory.create("chess")
pipeline = Pipeline(
    network=net,
    game="chess",
    train_batch_size=32,
    replay_buffer_max_size=500,
    mcts_batch_size=16,
    num_selfplay=2,
    num_rollout=32,
    iterations=1,
    verbose=True,
)

with cProfile.Profile() as profiler:
    pipeline.train("profile", None)

profiler.dump_stats("profile.prof")
print("Saved to profile.prof")