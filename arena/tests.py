import unittest

from arena.player import NetworkMCTSPlayer
from core import factory


class NetworkMCTSPlayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # factory caches by game/version, so this builds one network per run.
        cls.network = factory.build_network(game="chess")

    def test_network_is_played_in_eval_mode(self):
        # Training mode would let BatchNorm normalise each MCTS batch by its own
        # statistics, so a head-to-head result would depend on how leaves were
        # batched rather than on which network is stronger.
        self.network.train()

        player = NetworkMCTSPlayer(network=self.network, game="chess", num_rollout=1)

        self.assertFalse(player.mcts.network.training)


if __name__ == "__main__":
    unittest.main()
