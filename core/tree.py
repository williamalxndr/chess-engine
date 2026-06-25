import math
from abc import ABC, abstractmethod

import numpy as np
import torch

from core.network import PolicyValueNetwork, NetworkFactory
from game.rules import Rules, ChessRules, int_to_move
from core.node import Node


class BaseMCTS(ABC):
    """
    Base class for MCTS
    """

    def __init__(self, rules: Rules, num_rollout=1000, batch_size=50, verbose=True, seed=42):
        """
        Build the search tree and create the root node.

        Args:
            rules (Rules): game rules (legal moves, transitions, results).
            num_rollout (int): number of simulations to run per search().
            verbose (bool): whether to print log messages.
            seed (int): seed for the random number generator.
        """
        self.rules = rules
        self.num_rollout = num_rollout
        self.batch_size = batch_size
        self.verbose = verbose
        self.rng = np.random.default_rng(seed=seed)

        self.root = Node(self.rules, self.rules.base_state())

    def reset(self, state=None):
        if state is None:
            state = self.rules.base_state()
        self.root = Node(self.rules, state)

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
        Returns:
            (node, action):
            - action=None: node sudah punya priors, descend ke child terbaik
            tapi child belum di-materialize → expand child itu
            - action='evaluate': node belum punya priors → evaluate node ini dulu
            - action=<int>: node punya priors, child ini perlu di-materialize
        """
        node = self.root
        while self._has_been_expanded(node) and not node.get_leaf():
            action = self.best_action(node)
            child = node.children.get(action)
            if child is None:
                return node, action
            node = child
        
        if not self._has_been_expanded(node) and not node.get_leaf():
            return node, 'evaluate'
        
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
            node.untried_actions.remove(action)

        child_state = self.rules.transition_state(node.state, action)
        expanded_node = Node(self.rules, child_state, node, action)
        node.add_child(expanded_node, action)
        return expanded_node

    @abstractmethod
    def evaluate(self, nods: list[Node]):
        """
        Evaluate how good nodes are
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

    def collect_eval_nodes(self):
        """
        Run a search for `batch_size` times, and then 
        returns a batch of matrix to be evaluated from the network, to be backpropagated.
        
        ## Returns
            list[Nodes], int:
            Nodes to be evaluated in a single time, sims_done
        """
        eval_nodes = []
        sims_done = 0

        for _ in range(self.batch_size):
            selected_node, action = self.select()

            if action == 'evaluate':
                selected_node.apply_virtual_loss()
                eval_nodes.append(selected_node)

            elif selected_node.get_leaf():
                self.backpropagate(selected_node, selected_node.get_result())
                sims_done += 1

            else:
                node = self.expand(selected_node, action)
                if node.get_leaf():
                    self.backpropagate(node, node.get_result())
                    sims_done += 1
                else:
                    node.apply_virtual_loss()
                    eval_nodes.append(node)

        return eval_nodes, sims_done


    def search(self):
        self._apply_root_noise()
        
        sims_done = 0

        while sims_done < self.num_rollout:
            batch_nodes, batch_sims_done = self.collect_eval_nodes()
            sims_done += batch_sims_done

            if not batch_nodes:
                continue

            values = self.evaluate(batch_nodes)

            for node, value in zip(batch_nodes, values):
                node.remove_virtual_loss()
                if not node.children and not node.get_leaf():
                    child = self.expand(node)
                    self.backpropagate(child, value)
                else:
                    self.backpropagate(node, value)
                sims_done += 1

        return self.get_best_action()


    def get_child_visit_count(self) -> np.ndarray:
        """
        Visit counts of the root node's children, indexed by action.

        Returns:
            np.ndarray: array of length action_space_size; entry a is the
                visit count of the child reached by action a (0 if none).
        """
        counts = np.zeros(self.rules.action_space_size)
        for action, child in self.root.children.items():
            counts[action] = child._visit_count
        return counts

    def get_best_action(self):
        """
        Most-visited child action of the root node.

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

    def advance(self, action):
        """
        Move the root node along `action`

        Args:
            action (int): action to follow from the root node.

        Returns:
            is_leaf (bool): whether the new root node is terminal.
            result: game result at the new node (None if not terminal).

        Raises:
            ValueError: if `action` is illegal in the current state.
        """
        child = self.root.get_children_by_action(action)
        if child is not None:
            self.root = child
            self.root.parent = None
            return self.root.get_leaf(), self.root.get_result()
        if action in self.rules.get_legal_actions(self.root.state):
            self.root.add_child_by_action(action)
            return self.advance(action)
        raise ValueError("No child with that action")
    
    @abstractmethod
    def _apply_root_noise(self):
        """
        Mix exploration noise into the root's priors (defined by subclasses).
        """
        pass

    def get_current_state(self):
        """
        Return a copy of the root node's state.

        Returns:
            state: a copy of the current game state.
        """
        return self.root.state.copy()

    def log(self, msg):
        """
        Print `msg` only when verbose is enabled.

        Args:
            msg (str): message to print.
        """
        if self.verbose:
            print(msg)


class VanillaMCTS(BaseMCTS):
    def __init__(self, rules: Rules, num_rollout=1000,
                 exploration_constant=1.41, verbose=True, seed=42):
        """
        MCTS with UCT selection and random rollouts (no neural network).

        Args:
            rules (Rules): game rules.
            num_rollout (int): simulations per search().
            exploration_constant (float): UCT exploration weight (c).
            verbose (bool): whether to print log messages.
            seed (int): random number generator seed.
        """
        super().__init__(rules=rules, num_rollout=num_rollout, verbose=verbose, seed=seed)
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

    def evaluate(self, eval_nodes: list[Node]):
        """
        Estimate a node's value with a random rollout to the end of the game.

        Args:
            expanded_node (Node): node to evaluate.

        Returns:
            list[int]: the terminal result if the node is terminal, otherwise the
                result of a random rollout from it.
        """
        
        return [self.rules.rollout(node.state) for node in eval_nodes]

    def _apply_root_noise(self):
        """
        No-op: vanilla MCTS adds no root noise.
        """
        return


class NetworkMCTS(BaseMCTS):
    def __init__(self, network: PolicyValueNetwork,
                 num_rollout=1000, batch_size=50, 
                 c_puct=1.41, epsilon=0.25, seed=42, alpha=0.03,
                 add_noise=False, network_train=False, verbose=True):
        """
        MCTS guided by a policy-value network (AlphaZero-style PUCT).

        Args:
            network (PolicyValueNetwork): network producing priors and a value.
            rules (Rules): game rules.
            num_rollout (int): simulations per search().
            c_puct (float): PUCT exploration weight.
            epsilon (float): weight of the Dirichlet root noise (0 disables it).
            seed (int): random number generator seed.
            alpha (float): Dirichlet concentration parameter for root noise.
            add_noise (bool): if False, epsilon is forced to 0.
            network_train (bool): put the network in train() (True) or eval() mode.
            verbose (bool): whether to print log messages.
        """
        rules = network.rules
        super().__init__(rules=rules, num_rollout=num_rollout, batch_size=batch_size, verbose=verbose, seed=seed)
        self.device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
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
        prior = node.priors.get(action, 0)
        if node is not self.root:  
            return prior

        noise = node.noise.get(action, 0)
        return (1 - self.epsilon) * prior + self.epsilon * noise
    
    def forward_tensor(self, eval_encoded_states: torch.Tensor):
        if not isinstance(eval_encoded_states, torch.Tensor):
            raise ValueError("eval_encoded_states is not a torch.Tensor")
        
        if eval_encoded_states.ndim != 4:
            raise ValueError("eval_encoded_states shape must be (B x C x H x W)")
        
        return self.network(eval_encoded_states)
    
    def encode_nodes(self, eval_nodes: list[Node]):
        """
        Returns a tensor of (B x C x H x W) to be forwarded/evaluated by the network
        """
        encoded_eval_states = [self.rules.encode(node.state) for node in eval_nodes]      # Encode ensures that eval_nodes state to be shaped (C x H x W)
        return torch.stack(encoded_eval_states).to(self.device)       


    def forward_nodes(self, eval_nodes: list[Node]):
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
        batch = self.encode_nodes(eval_nodes)
        return self.forward_tensor(batch)
    
    def set_priors(self, node: Node, policy: np.ndarray):
        legal_actions = node.get_legal_actions()
        mask = np.ones(policy.shape, dtype=bool)
        mask[legal_actions] = False
        policy[mask] = -1e9 
        
        policy = policy - np.max(policy)  
        policy = np.exp(policy)
        policy = policy / np.sum(policy)
        
        node.priors = {a: float(policy[a]) for a in legal_actions}

    @torch.no_grad()
    def evaluate(self, eval_nodes: list[Node]) -> np.ndarray:
        """
        Run the network on eval_nodes, set priors on each, and return values.

        Args:
            eval_nodes (list[Node]): nodes to evaluate.

        Returns:
            np.ndarray: value estimates, shape (B,).
        """
        
        policy_head, value_head = self.forward_nodes(eval_nodes)

        policies = policy_head.cpu().numpy()  # (B, action_space_size)
        values = value_head.cpu().numpy()     # (B, 1)

        for i, node in enumerate(eval_nodes):
            self.set_priors(node, policies[i])

        return values[:, 0]  # (B,)
    
    def _apply_root_noise(self):
        """
        Add Dirichlet exploration noise to the root's priors.

        Evaluates the root first if it has no priors yet, then samples one
        Dirichlet vector over its legal actions. No-op when epsilon == 0.
        """
        # Add prior
        root = self.root
        if not root.priors:
            self.evaluate([root]) 

        # Add noise
        if self.epsilon == 0:
            return

        legal_actions = list(root.priors.keys())
        noise = self.rng.dirichlet([self.alpha] * len(legal_actions))
        root.noise = {a: float(noise[i]) for i, a in enumerate(legal_actions)}

    def enable_noise(self, epsilon=0.25, alpha=0.03):
        self.epsilon = epsilon
        self.alpha = alpha

    def disable_noise(self):
        self.epsilon = 0
        
            

if __name__ == "__main__":
    network = NetworkFactory.create("chess")
    net_agent = NetworkMCTS(network=network, num_rollout=30, batch_size=10)
    # vanilla_agent = VanillaMCTS(rules=ChessRules(), num_rollout=10)

    best_action_net = int_to_move(net_agent.search())
    # best_action_vanilla = int_to_move(vanilla_agent.search())

    print(best_action_net)

