import copy
import math
from abc import ABC, abstractmethod

import numpy as np
import torch

from core.network import PolicyValueNetwork
from game.env import Environment
from game.rules import Rules, TicTacToeRules
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

    def score(self, node: "Node"):
        """
        Scoring a node for the selection step.
        """
        return self.q(node) + self.u(node)   # mean value + exploration

    def q(self, node: "Node"):
        return node.q()

    @abstractmethod
    def u(self, node):
        pass

    def best_child(self, node: "Node"):
        """
        Determine the best child to descend to based on the scoring method.
        """
        return max(node.children, key=lambda ch: self.score(ch))

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

    @abstractmethod
    def expand(self, selected_node):
        """
        Expand the selected node with an untried action.

        Args:
            selected_node (Node): the node selected by the selection step

        Returns:
            Node: the newly expanded node
        """
        pass

    @abstractmethod
    def evaluate(self, node):
        """
        Evaluate the expanded node (rollout, or network value estimate).

        Args:
            node (Node): the node to evaluate

        Returns:
            value: 1 if win, -1 if lose, 0 if draw (or a network estimate)
        """
        pass

    def backpropagate(self, node: "Node", reward):
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
            selected_node = self.select()                         # Selection
            expanded_node = self.expand(selected_node)            # Expansion
            reward = self.evaluate(expanded_node)                 # Simulation
            self.backpropagate(expanded_node, reward)             # Backpropagation

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

    def u(self, node: "Node"):
        if node._visit_count == 0:
            return float("inf")
        return node.q() + self.exploration_constant * math.sqrt(
            np.log(node.parent._visit_count) / node._visit_count
        )

    def expand(self, selected_node: "Node"):
        if selected_node.get_leaf():
            return selected_node

        untried_action = selected_node.untried_actions.pop()
        child_state = self.rules.transition_state(selected_node.state, untried_action, selected_node.turn)
        expanded_node = Node(self.rules, child_state, selected_node, untried_action)
        selected_node.add_child(expanded_node)
        return expanded_node

    def evaluate(self, expanded_node: "Node"):
        if expanded_node.get_leaf():
            return expanded_node.result
        return self.rules.rollout(expanded_node.state)

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

    def u(self, node: "Node"):
        return self.c_puct * self.p(node) * math.sqrt(node.parent._visit_count) / (1 + node._visit_count)

    def p(self, node: "Node"):
        return (1 - self.epsilon) * node.prior + self.epsilon * node.noise

    def expand(self, selected_node: "Node"):
        if selected_node.get_leaf():
            return selected_node

        # For network MCTS, expand all children up front so priors can be assigned in one network call.
        while not selected_node.is_fully_expanded():
            untried_action = selected_node.untried_actions.pop()
            expanded_state = self.rules.transition_state(selected_node.state, untried_action, selected_node.turn)
            expanded_node = Node(self.rules, expanded_state, selected_node, untried_action)
            selected_node.add_child(expanded_node)

        return selected_node

    def evaluate(self, selected_node: "Node"):
        """
        Assigns priors to the children of `selected_node` and returns the
        network's value estimate for `selected_node` itself.
        """
        if selected_node.get_leaf():
            return selected_node.get_result()

        state = selected_node.state
        torch_state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0)
        policy_head, value_head = self.network(torch_state)

        policy_head = policy_head.squeeze(0)
        value_head = value_head.item()

        legal_actions = self.rules.get_legal_actions(state)
        mask = np.ones(policy_head.shape, dtype=bool)
        mask[legal_actions] = False
        policy_head[mask] = 0
        policy_head = policy_head / torch.sum(policy_head)

        for child in selected_node.children:
            child.prior = policy_head[child.action].item()

        return value_head

    def _apply_root_noise(self):
        if self.epsilon == 0:
            return

        root = self.observed

        if not root.is_fully_expanded():
            self.expand(root)
            self.evaluate(root)

        legal_actions = self.rules.get_legal_actions(root.state)
        noise = self.rng.dirichlet([self.alpha] * len(legal_actions))
        action_to_noise = {a: noise[i] for i, a in enumerate(legal_actions)}

        for child in root.children:
            child.noise = action_to_noise[child.action]


if __name__ == "__main__":
    network = PolicyValueNetwork()
    net_agent = NetworkMCTS(network=network, rules=TicTacToeRules(), num_rollout=1000)
    vanilla_agent = VanillaMCTS(rules=TicTacToeRules(), num_rollout=1000)

    best_action_net = net_agent.search()
    best_action_vanilla = vanilla_agent.search()

    print(best_action_net, best_action_vanilla)
    
