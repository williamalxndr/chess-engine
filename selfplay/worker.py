import numpy as np
import torch
import time
import sys

from core.tree import NetworkMCTS
from core.network import PolicyValueNetwork, NetworkFactory
from core.encoder import *
from core.node import Node


class GameWorker:
    def __init__(self, 
                 game: str, version: str = "latest",
                 file_name: str = None, parent_dir: str = "checkpoints", path: str = None,
                 network: PolicyValueNetwork = None,
                 seed: int = 42, num_rollout: int = 200, batch_size: int = 50, add_noise: bool = True):

        self.mcts = NetworkMCTS(
            game=game, version=version,
            file_name=file_name, parent_dir=parent_dir, path=path,
            network=network,
            num_rollout=num_rollout, batch_size=batch_size,
            seed=seed, add_noise=add_noise,
        )
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if add_noise:
            self.mcts.enable_noise()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.mcts.reset()

    def collect_eval_nodes(self):
        """
        Returns list[Node]
        """
        return self.mcts.collect_eval_nodes()
    
    def encode_nodes(self, nodes: list[Node]):
        return self.mcts.encode_nodes(nodes)
    
    def backpropagate(self, nodes: list[int], values: list[int]):
        if len(nodes) != len(values):
            raise ValueError("len(nodes) != len(values)")

        for node, value in zip(nodes, values):
            self.mcts.backpropagate(node, value)

    def set_priors(self, nodes: list[Node], policies: list[np.ndarray]):
        if len(nodes) != len(policies):
            raise ValueError("len(nodes) != len(policies)")
        
        for node, policy in zip(nodes, policies):
            self.mcts.set_priors(node, policy)

    def get_best_action(self):
        return self.mcts.get_best_action()
    
    def advance(self, action: int):
        return self.mcts.advance(action)
    
    def get_current_state(self):
        return self.mcts.get_current_state()
    
    def get_encoded_current_state(self):
        state = self.get_current_state() 
        return self.mcts.encoder.encode(state)

    def run_search(self):
        """
        Runs the search, nothing returned
        """
        self.mcts.search()

    def get_policy(self):
        """
        Returns the list of policy with size=action space size
        """
        return self.mcts.get_policy()
    
    def sample_action(self, policy):
        """
        Returns the action the agent decides from the policy
        """
        action = self.rng.choice(len(policy), p=policy)
        return action 
    
    def play_move(self):
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
        self.run_search()

        state = self.mcts.get_current_state()
        policy = self.get_policy()
        action = self.sample_action(policy)

        state = self.mcts.encoder.encode(state)
        policy = torch.tensor(policy)

        game_over, result = self.mcts.advance(action)
        return state, policy, game_over, result

    def run_game(self, display=True):
        self.reset()
        trajectory = []

        game_over = False
        first = True

        if display:
            board_str = str(self.mcts.root.state)
            sys.stdout.write(board_str + "\n")
            sys.stdout.flush()
            first = False 

        while not game_over:
            state, policy, game_over, result = self.play_move()
            trajectory.append((state, policy))

            if display:
                board_str = str(self.mcts.root.state)
                if not first:
                    sys.stdout.write(f"\033[{board_str.count(chr(10)) + 1}A\033[0J")
                sys.stdout.write(board_str + "\n")
                sys.stdout.flush()
                first = False

        if display:
            print(self.mcts.root.state.result())
            fen = self.mcts.root.state.fen()
            print(f"Board: https://lichess.org/editor/{fen.replace(' ', '_')}")
            
        return trajectory, result        
        

if __name__ == "__main__":
    worker = GameWorker(game="chess", seed=20, num_rollout=100, batch_size=40)

    for _ in range(1):
        print(worker.run_game())