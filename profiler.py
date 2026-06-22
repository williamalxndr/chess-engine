import cProfile
import pstats

from training.pipeline import Pipeline
from core.network import NetworkFactory


net = NetworkFactory.create("chess")
pipeline = Pipeline(net, "chess", verbose=True)

with cProfile.Profile() as profiler:
    pipeline.train("profile", 0.05)

stats = pstats.Stats(profiler)
stats.sort_stats(pstats.SortKey.TIME).print_stats(10)
stats.sort_stats(pstats.SortKey.TIME).print_stats("cpu")
stats.print_callers("cpu")  