import copy
import math
from abc import ABC, abstractmethod

import numpy as np
import torch

from core.network import PolicyValueNetwork
from game.env import Environment
from game.rules import Rules, TicTacToeRules, ChessRules, int_to_move
from core.node import Node


class BaseMCTS(ABC):
    """
    Game-agnostic MCTS skeleton. Depends only on a `Rules` instance for
    search; an `Environment` is optional and only used to keep a live game
    session in sync when `advance()` is called with `do_step=True`.
    """

    def __init__(self, rules: Rules, env: Environment = None, num_rollout=1000, verbose=True, seed=42):
        self.rules = rules
        self.env = env
        self.num_rollout = num_rollout
        self.verbose = verbose
        self.rng = np.random.default_rng(seed=seed)

        self.root = Node(self.rules, self._initial_state())
        self.observed = self.root

    def _initial_state(self):
        return self.env.state if self.env is not None else self.rules.base_state()

    def reset(self, state=None):
        """
        Clears the stored tree and starts over from `state` (or the live
        env's state, or the game's base state, in that order of priority).
        """
        if self.env is not None:
            self.env.reset(state)
            state = self.env.state
        elif state is None:
            state = self.rules.base_state()

        self.root = Node(self.rules, state)
        self.observed = self.root

    def set_env(self, env: Environment):
        self.env = env
        self.reset()

    def score(self, node: Node, action: int):
        """
        Scoring a node for the selection step.
        """
        return self.q(node, action) + self.u(node, action)   # mean value (q) + exploration (u)

    def q(self, node: Node, action: int):
        """
        Q(S, A)
        """
        child = node.get_children_by_action(action)
        if child is None:
            return 0
        return child.q()

    @abstractmethod
    def u(self, node: Node, action: int):
        """
        U(S, A)
        """
        pass

    def best_child(self, node: Node):
        """
        Determine the best child to descend to based on the scoring method.
        """
        legal_actions = node.get_legal_actions()
        best_action = max(legal_actions, key=lambda a: self.score(node, a))
        child = node.children.get(best_action)

        if child is None:
            child = self.expand(node, best_action)

        return child

    def select(self):
        """
        Descend the tree picking the best child until an unexpanded or
        terminal node is reached.

        Returns:
            Node: the node to expand next
        """
        selected_node = self.observed
        while selected_node.is_fully_expanded() and not selected_node.get_leaf():
            selected_node = self.best_child(selected_node)
        return selected_node
    
    def expand(self, node: Node, action=None):
        if node.get_leaf():
            return node
        
        if action is None:
            action = node.get_untried_action()
        
        child_state = self.rules.transition_state(node.state, action)
        expanded_node = Node(self.rules, child_state, node)
        node.add_child(expanded_node, action)

        return expanded_node

        
    @abstractmethod
    def evaluate(self, node: Node):
        """
        Evaluate how good a node is

        Args:
            node (Node)

        Returns:
            int: absolute value (-1 for the first player to move, 1 for the second player to move)
        """
        pass

    @abstractmethod
    def expand_and_evaluate(self, selected_node):
        """
        Expand one child then evaluate how good is that node

        Args:
            selected_node (Node)

        Returns:
            (node_to_backpropagate_from, reward)
        """
        pass


    def backpropagate(self, node: Node, reward):
        """
        Update the value of every node on the path back to the root.

        Args:
            reward: reward from the simulation step.
        """
        while node is not None:
            node.update(reward)
            node = node.parent

    def search(self):
        """
        Run `num_rollout` simulations and return the best action found.
        """
        self._apply_root_noise()

        for _ in range(self.num_rollout):
            selected_node = self.select()                               # Selection
            node, reward = self.expand_and_evaluate(selected_node)      # Expansion and evaluation
            self.backpropagate(node, reward)                            # Backpropagation

        return self.get_best_action()

    def get_child_visit_count(self) -> np.ndarray:
        """
        Returns the visit-count array for the currently observed node.
        """
        counts = np.zeros(self.rules.action_space_size)
        for ch in self.observed.children:
            counts[ch.action] = ch._visit_count
        return counts

    def get_best_action(self):
        return int(np.argmax(self.get_child_visit_count()))

    def get_policy(self) -> np.ndarray:
        """
        Returns the visit-count distribution over actions, normalized to
        sum to 1. E.g. [0.1, 0.2, 0, 0.15, 0, 0.05, 0.15, 0, 0.35].
        """
        counts = self.get_child_visit_count()
        total = counts.sum()
        if total == 0:
            raise ValueError(
                "Children have no visits at all. Run search() first, or if "
                "you already have, increase num_rollout (must exceed the "
                "number of legal actions)."
            )
        return counts / total

    def advance(self, action, do_step=True):
        """
        Move `self.observed` forward by `action`, expanding the tree if
        needed. If an `env` is attached and `do_step` is True, the live
        session is advanced too.

        Returns:
            is_leaf, result
            is_leaf (bool): True if the resulting state is terminal
            result (None/int): None if the game is still going, otherwise
                the result (-1/1/0)
        """
        for child in self.observed.children:
            if child.action == action:
                if do_step and self.env is not None:
                    self.env.step(action)
                self.observed = child
                return self.observed.get_leaf(), self.observed.get_result()

        if action in self.rules.get_legal_actions(self.observed.state):
            self.observed.add_child_by_action(action)
            return self.advance(action, do_step)

        raise ValueError("No child with that action")

    @abstractmethod
    def _apply_root_noise(self):
        pass

    def get_current_state(self):
        """
        Returns a copy of the currently observed state.
        """
        return self.observed.state.copy()

    def log(self, msg):
        if self.verbose:
            print(msg)


class VanillaMCTS(BaseMCTS):
    def __init__(self, rules: Rules, env: Environment = None, num_rollout=1000,
                 exploration_constant=1.41, verbose=True, seed=42):
        super().__init__(rules, env, num_rollout, verbose, seed)
        self.exploration_constant = exploration_constant

    def u(self, node: Node, action: int):
        child = node.get_children_by_action(action)
        if child is None or child._visit_count == 0:
            return float("inf")
        
        return self.exploration_constant * math.sqrt(
            np.log(node._visit_count) / child._visit_count
        )

    def evaluate(self, expanded_node: Node):
        if expanded_node.get_leaf():
            return expanded_node.result
        return self.rules.rollout(expanded_node.state)
    
    def expand_and_evaluate(self, selected_node):
        expanded_node = self.expand(selected_node)
        value = self.evaluate(expanded_node)
        return expanded_node, value

    def _apply_root_noise(self):
        return


class NetworkMCTS(BaseMCTS):
    def __init__(self, network: PolicyValueNetwork, rules: Rules, env: Environment = None,
                 num_rollout=1000, c_puct=1.41, epsilon=0.25, seed=42, alpha=0.03,
                 add_noise=False, network_train=False, verbose=True):
        super().__init__(rules, env, num_rollout, verbose, seed)
        self.network = network
        self.network.train(network_train)
        self.c_puct = c_puct
        self.epsilon = epsilon if add_noise else 0
        self.alpha = alpha

    def u(self, node: Node, action: int):
        child = node.get_children_by_action(action)
        return self.c_puct * self.p(node) * math.sqrt(node.parent._visit_count) / (1 + node._visit_count)

    def p(self, node: Node):
        return (1 - self.epsilon) * node.prior + self.epsilon * node.noise
    
    def forward_network(self, node: Node):
        """
        Returns policy_head, value_head
        """
        encoded_state = self.rules.encode(node.state)
        torch_state = torch.from_numpy(encoded_state)
        return self.network(torch_state)

    def evaluate(self, node: Node):
        if node.get_leaf():
            return node.get_result()

        policy_head, value_head = self.forward_network(node)

        legal_actions = node.get_legal_actions()
        
        # masking (0 for the illegal action)
        mask = np.ones(policy_head.shape, dtype=bool)
        mask[legal_actions] = False
        policy_head[mask] = 0
        policy_head = policy_head / torch.sum(policy_head)

        node.priors = {a: policy_head[a].item() for a in legal_actions}

        return value_head
        
    def expand_and_evaluate(self, node: Node):
        if node.get_leaf():
            return node, node.ge_result()
        
        reward = self.evaluate(node)

        priors = node.priors
        best_action = max(priors, key=priors.get)

        expanded_node = self.expand(node, best_action)

        return expanded_node, reward
        

    def _apply_root_noise(self):
        if self.epsilon == 0:
            return

        root = self.observed

        if not root.priors:
            self.evaluate(root)

        legal_actions = list(root.priors.keys())
        noise = self.rng.dirichlet([self.alpha] * len(legal_actions))
        root.noise = {a: noise[i] for i, a in enumerate(legal_actions)}

if __name__ == "__main__":
    network = PolicyValueNetwork(rules=ChessRules())
    net_agent = NetworkMCTS(network=network, rules=ChessRules(), num_rollout=1000)
    vanilla_agent = VanillaMCTS(rules=ChessRules(), num_rollout=1000)

    best_action_net = int_to_move(net_agent.search())
    best_action_vanilla = vanilla_agent.search()

    print(best_action_net, best_action_vanilla)
    
