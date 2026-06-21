import math
from abc import ABC, abstractmethod

import numpy as np
import torch

from core.network import PolicyValueNetwork
from game.env import Environment
from game.rules import Rules, ChessRules, int_to_move
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
        """Q(S, A)"""
        child = node.children.get(action)
        if child is None:
            return 0
        return child.q()

    @abstractmethod
    def u(self, node: Node, action: int):
        """U(S, A)"""
        pass

    def best_action(self, node: Node):
        legal_actions = node.get_legal_actions()
        return max(legal_actions, key=lambda a: self.score(node, a))

    @abstractmethod
    def _has_been_expanded(self, node: Node) -> bool:
        """
        Whether `node` is "ready" for PUCT-based descent into its children
        (VanillaMCTS: all untried actions consumed. NetworkMCTS: node has
        been evaluated by the network, so it has priors).
        """
        pass

    def select(self):
        """
        Descend the tree.

        Returns:
            (node, action): the edge that needs expand_and_evaluate().
            `action` is None if `node` itself is what needs evaluating
            (a fresh/never-evaluated node, or a terminal node).
        """
        node = self.observed
        while self._has_been_expanded(node) and not node.get_leaf():
            action = self.best_action(node)
            child = node.children.get(action)
            if child is None:
                return node, action
            node = child
        return node, None

    def expand(self, node: Node, action=None):
        """
        Materialize ONE child for `action` (or an arbitrary untried action
        if none given) and attach it to `node`.
        """
        if node.get_leaf():
            return node
        
        if action is None:
            action = node.get_untried_action()
        else:
            node.untried_actions = [a for a in node.untried_actions if a != action]

        child_state = self.rules.transition_state(node.state, action)
        expanded_node = Node(self.rules, child_state, node, action)
        node.add_child(expanded_node, action)
        return expanded_node

    @abstractmethod
    def evaluate(self, node: Node):
        """
        Evaluate how good a node is.

        Returns:
            float: absolute value
        """
        pass

    @abstractmethod
    def expand_and_evaluate(self, node: Node, action):
        """
        Args:
            node (Node): node returned by select()
            action (int or None): edge to materialize, or None if `node`
                itself is what needs evaluating

        Returns:
            (node_to_backpropagate_from, reward)
        """
        pass

    def backpropagate(self, node: Node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent

    def search(self):
        self._apply_root_noise()

        for _ in range(self.num_rollout):
            selected_node, action = self.select()
            node, reward = self.expand_and_evaluate(selected_node, action)
            self.backpropagate(node, reward)

        return self.get_best_action()

    def get_child_visit_count(self) -> np.ndarray:
        counts = np.zeros(self.rules.action_space_size)
        for action, child in self.observed.children.items():
            counts[action] = child._visit_count
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
        child = self.observed.get_children_by_action(action)
        if child is not None:
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
        return self.observed.state.copy()

    def log(self, msg):
        if self.verbose:
            print(msg)


class VanillaMCTS(BaseMCTS):
    def __init__(self, rules: Rules, env: Environment = None, num_rollout=1000,
                 exploration_constant=1.41, verbose=True, seed=42):
        super().__init__(rules, env, num_rollout, verbose, seed)
        self.exploration_constant = exploration_constant

    def _has_been_expanded(self, node: Node) -> bool:
        return node.is_fully_expanded()

    def u(self, node: Node, action: int):
        child = node.get_children_by_action(action)
        
        if child is None or child._visit_count == 0:
            return float("inf")
        
        return self.exploration_constant * math.sqrt(
            math.log(node._visit_count) / child._visit_count
        )

    def evaluate(self, expanded_node: Node):
        if expanded_node.get_leaf():
            return expanded_node.result
        return self.rules.rollout(expanded_node.state)

    def expand_and_evaluate(self, node: Node, action):
        expanded_node = self.expand(node, action)
        return expanded_node, self.evaluate(expanded_node)

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

    def _has_been_expanded(self, node: Node) -> bool:
        return bool(node.priors)

    def u(self, node: Node, action: int):
        child = node.children.get(action)
        n_action = child._visit_count if child is not None else 0
        return self.c_puct * self.p(node, action) * math.sqrt(node._visit_count) / (1 + n_action)

    def p(self, node: Node, action: int):
        prior = node.priors.get(action, 0)
        noise = node.noise.get(action, 0)
        return (1 - self.epsilon) * prior + self.epsilon * noise

    def forward_network(self, node: Node):
        """Returns policy_head, value_head"""
        encoded_state = self.rules.encode(node.state)
        torch_state = torch.from_numpy(encoded_state).float()
 
        if torch_state.ndim == 2:          # (H, W) -> single channel, single state
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
        elif torch_state.ndim == 3:        # (C, H, W) -> single state
            torch_state = torch_state.unsqueeze(0)
        elif torch_state.ndim == 4:        # (B, C, H, W) -> already batched
            pass
        else:
            raise ValueError(
                f"encode() returned unexpected shape {tuple(torch_state.shape)} -- "
                f"expected (H,W), (C,H,W), or (B,C,H,W)"
            )
 
        return self.network(torch_state)
    def evaluate(self, node: Node):
        if node.get_leaf():
            return node.get_result()

        policy_head, value_head = self.forward_network(node)
        policy_head = policy_head.squeeze(0)
        value_head = value_head.item()

        legal_actions = node.get_legal_actions()
        mask = np.ones(policy_head.shape, dtype=bool)
        mask[legal_actions] = False
        policy_head[mask] = 0
        policy_head = policy_head / torch.sum(policy_head)

        node.priors = {a: policy_head[a].item() for a in legal_actions}
        return value_head

    def expand_and_evaluate(self, node: Node, action):
        if action is not None:
            node = self.expand(node, action)

        if node.get_leaf():
            return node, node.get_result()

        reward = self.evaluate(node)
        return node, reward

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
    net_agent = NetworkMCTS(network=network, rules=ChessRules(), num_rollout=100)
    vanilla_agent = VanillaMCTS(rules=ChessRules(), num_rollout=10)

    best_action_net = int_to_move(net_agent.search())
    best_action_vanilla = int_to_move(vanilla_agent.search())

    print(best_action_net, best_action_vanilla)
    
