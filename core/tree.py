import math
from abc import ABC, abstractmethod

import numpy as np
import torch

from core.network import PolicyValueNetwork, NetworkFactory
from game.env import Environment
from game.rules import Rules, ChessRules, int_to_move
from core.node import Node


class BaseMCTS(ABC):
    """
    Base class for MCTS
    """

    def __init__(self, rules: Rules, env: Environment = None, num_rollout=1000, verbose=True, seed=42):
        """
        Build the search tree and create the root node.

        Args:
            rules (Rules): game rules (legal moves, transitions, results).
            env (Environment): live game session, or None for a standalone tree.
            num_rollout (int): number of simulations to run per search().
            verbose (bool): whether to print log messages.
            seed (int): seed for the random number generator.
        """
        self.rules = rules
        self.env = env
        self.num_rollout = num_rollout
        self.verbose = verbose
        self.rng = np.random.default_rng(seed=seed)

        self.root = Node(self.rules, self._initial_state())
        self.observed = self.root

    def _initial_state(self):
        """
        Pick the state the root node starts from.

        Returns:
            state: the env's current state if an env is attached, otherwise
                the rules' base state.
        """
        return self.env.state if self.env is not None else self.rules.base_state()

    def reset(self, state=None):
        """
        Clear the stored tree and start over from `state`.

        Args:
            state: state to restart from. If an env is attached the env is reset to it, if None, the rules' base state is used.
        """
        if self.env is not None:
            self.env.reset(state)
            state = self.env.state
        elif state is None:
            state = self.rules.base_state()

        self.root = Node(self.rules, state)
        self.observed = self.root

    def set_env(self, env: Environment):
        """
        Attach a live environment and reset the tree to its state.

        Args:
            env (Environment): the game session to attach.
        """
        self.env = env
        self.reset()

    def score(self, node: Node, action: int):
        """
        Selection score of an edge: exploitation (q) plus exploration (u).

        Args:
            node (Node): the parent node being expanded from.
            action (int): the edge (action) to score.

        Returns:
            float: combined score used to pick the next action.
        """
        return self.q(node, action) + self.u(node, action)   # mean value (q) + exploration (u)

    def q(self, node: Node, action: int):
        """
        Mean value Q(S, A) of taking `action` from `node`.

        Args:
            node (Node): the parent node.
            action (int): the edge (action) to evaluate.

        Returns:
            float: the child's mean value, or 0 if the child does not exist yet.
        """
        child = node.children.get(action)
        if child is None:
            return 0
        return child.q()

    @abstractmethod
    def u(self, node: Node, action: int):
        """
        Exploration bonus U(S, A) for an edge (defined by subclasses).

        Args:
            node (Node): the parent node.
            action (int): the edge (action) to evaluate.

        Returns:
            float: the exploration term added to q() during selection.
        """
        pass

    def best_action(self, node: Node):
        """
        Return the legal action with the highest selection score.

        Args:
            node (Node): node to pick an action from.

        Returns:
            int: the best-scoring legal action.
        """
        legal_actions = node.get_legal_actions()
        return max(legal_actions, key=lambda a: self.score(node, a))

    @abstractmethod
    def _has_been_expanded(self, node: Node) -> bool:
        """
        Whether `node` has already been expanded/evaluated once.

        Args:
            node (Node): node to check.

        Returns:
            bool: True if the node has the children/priors needed to descend.
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
        Materialize ONE child of `node` and attach it.

        Args:
            node (Node): node to expand from.
            action (int): edge to materialize. If None, an arbitrary untried
                action is popped instead.

        Returns:
            Node: the newly created child, or `node` itself if it is terminal.
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
        Evaluate how good a node is (defined by subclasses).

        Args:
            node (Node): node to evaluate.

        Returns:
            float: absolute value of the node.
        """
        pass

    @abstractmethod
    def expand_and_evaluate(self, node: Node, action):
        """
        Expand the selected edge (if any) and evaluate the resulting node.

        Args:
            node (Node): node returned by select().
            action (int): edge to materialize, or None if `node` itself is
                what needs evaluating.

        Returns:
            node (Node): the node to backpropagate from.
            reward (float): value to backpropagate.
        """
        pass

    def backpropagate(self, node: Node, reward):
        """
        Propagate `reward` from `node` up to the root.

        Args:
            node (Node): node to start backpropagation from.
            reward (float): absolute value to add along the path to the root.
        """
        while node is not None:
            node.update(reward)
            node = node.parent

    def search(self):
        """
        Run `num_rollout` simulations from the observed node and pick a move.

        Returns:
            int: the best action by visit count after searching.
        """
        self._apply_root_noise()

        for _ in range(self.num_rollout):
            selected_node, action = self.select()
            node, reward = self.expand_and_evaluate(selected_node, action)
            self.backpropagate(node, reward)

        return self.get_best_action()

    def get_child_visit_count(self) -> np.ndarray:
        """
        Visit counts of the observed node's children, indexed by action.

        Returns:
            np.ndarray: array of length action_space_size; entry a is the
                visit count of the child reached by action a (0 if none).
        """
        counts = np.zeros(self.rules.action_space_size)
        for action, child in self.observed.children.items():
            counts[action] = child._visit_count
        return counts

    def get_best_action(self):
        """
        Most-visited child action of the observed node.

        Returns:
            int: the action with the highest visit count.
        """
        return int(np.argmax(self.get_child_visit_count()))

    def get_policy(self) -> np.ndarray:
        """
        Visit-count distribution over actions, normalized to sum to 1.

        E.g. [0.1, 0.2, 0, 0.15, 0, 0.05, 0.15, 0, 0.35].

        Returns:
            np.ndarray: probability for each action (length action_space_size).

        Raises:
            ValueError: if no child has been visited yet.
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
        Move the observed node along `action`, optionally stepping the env.

        Args:
            action (int): action to follow from the observed node.
            do_step (bool): if True and an env is attached, also step the env.

        Returns:
            is_leaf (bool): whether the new observed node is terminal.
            result: game result at the new node (None if not terminal).

        Raises:
            ValueError: if `action` is illegal in the current state.
        """
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
        """
        Mix exploration noise into the root's priors (defined by subclasses).
        """
        pass

    def get_current_state(self):
        """
        Return a copy of the observed node's state.

        Returns:
            state: a copy of the current game state.
        """
        return self.observed.state.copy()

    def log(self, msg):
        """
        Print `msg` only when verbose is enabled.

        Args:
            msg (str): message to print.
        """
        if self.verbose:
            print(msg)


class VanillaMCTS(BaseMCTS):
    def __init__(self, rules: Rules, env: Environment = None, num_rollout=1000,
                 exploration_constant=1.41, verbose=True, seed=42):
        """
        MCTS with UCT selection and random rollouts (no neural network).

        Args:
            rules (Rules): game rules.
            env (Environment): live game session, or None.
            num_rollout (int): simulations per search().
            exploration_constant (float): UCT exploration weight (c).
            verbose (bool): whether to print log messages.
            seed (int): random number generator seed.
        """
        super().__init__(rules, env, num_rollout, verbose, seed)
        self.exploration_constant = exploration_constant

    def _has_been_expanded(self, node: Node) -> bool:
        """
        A node counts as expanded once all its actions have been tried.

        Args:
            node (Node): node to check.

        Returns:
            bool: True if the node is fully expanded.
        """
        return node.is_fully_expanded()

    def u(self, node: Node, action: int):
        """
        UCT exploration term: c * sqrt(ln(N_parent) / N_child).

        Args:
            node (Node): the parent node.
            action (int): edge to evaluate.

        Returns:
            float: exploration bonus; inf if the child is unvisited.
        """
        child = node.get_children_by_action(action)

        if child is None or child._visit_count == 0:
            return float("inf")

        return self.exploration_constant * math.sqrt(
            math.log(node._visit_count) / child._visit_count
        )

    def evaluate(self, expanded_node: Node):
        """
        Estimate a node's value with a random rollout to the end of the game.

        Args:
            expanded_node (Node): node to evaluate.

        Returns:
            int: the terminal result if the node is terminal, otherwise the
                result of a random rollout from it.
        """
        if expanded_node.get_leaf():
            return expanded_node.result
        return self.rules.rollout(expanded_node.state)

    def expand_and_evaluate(self, node: Node, action):
        """
        Expand the edge and score the new child via rollout.

        Args:
            node (Node): node returned by select().
            action (int): edge to materialize.

        Returns:
            expanded_node (Node): the newly created child.
            reward (int): rollout/terminal value of that child.
        """
        expanded_node = self.expand(node, action)
        return expanded_node, self.evaluate(expanded_node)

    def _apply_root_noise(self):
        """
        No-op: vanilla MCTS adds no root noise.
        """
        return


class NetworkMCTS(BaseMCTS):
    def __init__(self, network: PolicyValueNetwork, rules: Rules, env: Environment = None,
                 num_rollout=1000, c_puct=1.41, epsilon=0.25, seed=42, alpha=0.03,
                 add_noise=False, network_train=False, verbose=True):
        """
        MCTS guided by a policy-value network (AlphaZero-style PUCT).

        Args:
            network (PolicyValueNetwork): network producing priors and a value.
            rules (Rules): game rules.
            env (Environment): live game session, or None.
            num_rollout (int): simulations per search().
            c_puct (float): PUCT exploration weight.
            epsilon (float): weight of the Dirichlet root noise (0 disables it).
            seed (int): random number generator seed.
            alpha (float): Dirichlet concentration parameter for root noise.
            add_noise (bool): if False, epsilon is forced to 0.
            network_train (bool): put the network in train() (True) or eval() mode.
            verbose (bool): whether to print log messages.
        """
        super().__init__(rules, env, num_rollout, verbose, seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = network.to(self.device)
        self.network.train(network_train)
        self.c_puct = c_puct
        self.epsilon = epsilon if add_noise else 0
        self.alpha = alpha

    def _has_been_expanded(self, node: Node) -> bool:
        """
        A node is expanded once the network has filled in its priors.

        Args:
            node (Node): node to check.

        Returns:
            bool: True if the node has priors.
        """
        return bool(node.priors)

    def u(self, node: Node, action: int):
        """
        PUCT exploration term: c_puct * P(s,a) * sqrt(N_parent) / (1 + N_child).

        Args:
            node (Node): the parent node.
            action (int): edge to evaluate.

        Returns:
            float: exploration bonus for the edge.
        """
        child = node.children.get(action)
        n_action = child._visit_count if child is not None else 0
        return self.c_puct * self.p(node, action) * math.sqrt(node._visit_count) / (1 + n_action)

    def p(self, node: Node, action: int):
        """
        Prior probability of an edge, with Dirichlet noise.

        Args:
            node (Node): the parent node.
            action (int): edge to evaluate.

        Returns:
            float: (1 - epsilon) * prior + epsilon * noise.
        """
        prior = node.priors.get(action, 0)
        noise = node.noise.get(action, 0)
        return (1 - self.epsilon) * prior + self.epsilon * noise

    def forward_network(self, node: Node):
        """
        Run the network on a node's state, adding a batch dimension as needed.

        Args:
            node (Node): node whose state is encoded and fed to the network.

        Returns:
            policy_head (torch.Tensor): action priors, shape (1, action_space_size).
            value_head (torch.Tensor): scalar value estimate, shape (1, 1).

        Raises:
            ValueError: if encode() returns an unexpected shape.
        """
        encoded_state = self.rules.encode(node.state)
        torch_state = torch.from_numpy(encoded_state).float().to(self.device)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(device)

        if torch_state.ndim == 2:          # (H, W) -> single channel, single state
            torch_state = torch_state.unsqueeze(0).unsqueeze(0)
        elif torch_state.ndim == 3:        # (C, H, W) -> single state
            torch_state = torch_state.unsqueeze(0)
        elif torch_state.ndim == 4:        # (B, C, H, W) -> already batched
            pass
        else:
            raise ValueError(f"encode() returned unexpected shape {tuple(torch_state.shape)}, expected (H,W), (C,H,W), or (B,C,H,W)")
        
        return self.network(torch_state)
    
    def evaluate(self, node: Node):
        """
        Store network priors on the node and return its value estimate.

        Illegal moves are masked out and the remaining priors renormalized
        before being saved on the node.

        Args:
            node (Node): node to evaluate.

        Returns:
            float: the terminal result if the node is terminal, otherwise the
                network's value estimate.
        """
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
        """
        Expand the edge (if any), then evaluate the resulting node.

        Args:
            node (Node): node returned by select().
            action (int): edge to materialize, or None to evaluate `node` itself.

        Returns:
            node (Node): the node that was evaluated.
            reward (float): terminal result or network value of that node.
        """
        if action is not None:
            node = self.expand(node, action)

        if node.get_leaf():
            return node, node.get_result()

        reward = self.evaluate(node)
        return node, reward

    def _apply_root_noise(self):
        """
        Add dirichlet exploration noise to the root's priors.

        Evaluates the root first if it has no priors yet, then samples one
        Dirichlet vector over its legal actions. No-op when epsilon == 0.
        """
        if self.epsilon == 0:
            return

        root = self.observed
        if not root.priors:
            self.evaluate(root)

        legal_actions = list(root.priors.keys())
        noise = self.rng.dirichlet([self.alpha] * len(legal_actions))
        root.noise = {a: noise[i] for i, a in enumerate(legal_actions)}


if __name__ == "__main__":
    network = NetworkFactory.create("chess")
    net_agent = NetworkMCTS(network=network, rules=ChessRules(), num_rollout=100)
    vanilla_agent = VanillaMCTS(rules=ChessRules(), num_rollout=10)

    best_action_net = int_to_move(net_agent.search())
    best_action_vanilla = int_to_move(vanilla_agent.search())

    print(best_action_net, best_action_vanilla)

