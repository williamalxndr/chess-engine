import numpy as np

from core.tree import NetworkMCTS
from core.network import PolicyValueNetwork


class Generator:
    def __init__(self, mcts: NetworkMCTS, seed=42):
        self.mcts = mcts
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.mcts.reset()

    def _run_search(self):
        """
        Runs the search, nothing returned
        """
        self.mcts.search()

    def _get_policy(self):
        """
        Returns the list of policy with size=9 
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
            state (np.ndarray): The state of the current board
            policy (list): The policy after running the search
            game_over (bool): True if game has winner or drawn
            result (None/int): None if game still going, or return the winner (-1/1/0)
        """
        self._run_search()

        state = self.mcts.get_current_state()
        policy = self._get_policy()
        action = self._sample_policy(policy)

        game_over, result = self.mcts.advance(action)

        return state, policy, game_over, result

    def generate(self):
        """
        Reset the game, then
        MCTS plays 1 game until gets a result.

        ## Returns:
            trajectory, result
            trajectory (list): a list contains [s_0, pi_0, s_1, pi_1, ..., s_t-1, pi_t-1]
            result (int): game results, -1 if X wins, 1 if O wins, 0 if draw
        """
        self.reset()
        trajectory = []

        game_over = False
        while not game_over:
            state, policy, game_over, result = self.make_move()
            state = self.mcts.rules.encode(state)
            trajectory.append((state, policy))

        return trajectory, result
        
        

