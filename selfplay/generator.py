import numpy as np
import torch
import time
import sys

from core.tree import NetworkMCTS
from core.network import PolicyValueNetwork, NetworkFactory
from game.encoder import *


class Generator:
    def __init__(self, mcts: NetworkMCTS, add_noise=True, seed=42):
        self.mcts = mcts
        self.rng = np.random.default_rng(seed)

        if add_noise:
            self.mcts.enable_noise()

    def reset(self):
        self.mcts.reset()

    def _run_search(self):
        """
        Runs the search, nothing returned
        """
        self.mcts.search()

    def _get_policy(self):
        """
        Returns the list of policy with size=action space size
        """
        return self.mcts.get_policy()
    
    def _sample_policy(self, policy):
        """
        Returns the action the agent decides from the policy
        """
        action = self.rng.choice(len(policy), p=policy)
        return action 
    
    def make_move(self):
        """
        Run a search, decides an action, then
        Make a move then advance the board after performing the action

        Returns:
            state, policy, game_over, result
            state (torch.Tensor): The state of the current board
            policy (torch.Tensor): The policy after running the search
            game_over (bool): True if game has winner or drawn
            result (None/int): None if game still going, or return the winner (-1/1/0)
        """
        self._run_search()

        state = self.mcts.get_current_state()
        policy = self._get_policy()
        action = self._sample_policy(policy)

        state = self.mcts.rules.encode(state)
        policy = torch.tensor(policy)

        game_over, result = self.mcts.advance(action)
        return state, policy, game_over, result

    def generate(self, display=False):
        self.reset()
        trajectory = []

        game_over = False
        first = True

        while not game_over:
            state, policy, game_over, result = self.make_move()
            trajectory.append((state, policy))

            if display:
                board_str = str(decode_chess_state(state))
                if not first:
                    sys.stdout.write(f"\033[{board_str.count(chr(10)) + 1}A\033[0J")
                sys.stdout.write(board_str + "\n")
                sys.stdout.flush()
                first = False

        if display:
            print(self.mcts.observed.state.result())
        return trajectory, result        
        

if __name__ == "__main__": 
    mcts = NetworkMCTS(NetworkFactory.create("chess"), num_rollout=100, batch_size=10)
    generator = Generator(mcts)

    for _ in range(5):
        generator.generate()

