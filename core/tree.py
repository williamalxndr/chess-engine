from game.env import TicTacToe
from core.network import PolicyValueNetwork

import numpy as np
import torch
import math
import copy
from abc import ABC, abstractmethod

BASE_BOARD = TicTacToe.base_state()

class BaseMCTS(ABC):
    def __init__(self, env: TicTacToe = None, num_rollout=1000, verbose=True, seed=42):
        self.env = env if env is not None else TicTacToe()
        self.num_rollout = num_rollout
        self.root = Node(self.env.board)
        self.observed = self.root
        self.verbose = verbose
        self.rng = np.random.default_rng(seed=seed)

    def reset(self):
        """
        Clears the stored node
        """
        self.env.reset()
        self.root = Node(self.env.board)
        self.observed = self.root

    def set_env(self, env: TicTacToe):
        self.env = env
        self.reset()

    def score(self, node: "Node"):
        """
        Scoring a node for selection process
        """
        return self.q(node) + self.u(node)   # mean value + exploration
    
    def q(self, node: "Node"):
        return node.q()
    
    @abstractmethod
    def u(self, node):
        return NotImplemented 
    
    def best_child(self, node: "Node"):
        """
        Determining the best child to descend to based on the scoring method
        """
        return max(node.children, key=lambda ch: self.score(ch))

    def select(self):
        """
        Select the best child, if no child are already expanded
        Returns:
            Node: the best child node to be expanded
        """
        selected_node = self.observed
        while selected_node.is_fully_expanded() and not selected_node.is_leaf:
            selected_node = self.best_child(selected_node)

        return selected_node
    
    @abstractmethod
    def expand(self, selected_node: "Node"):
        """
        Expand the selected node with an untried action

        Args:
            selected_node (Node): The node that is selected by selection part

        Returns:
            expanded_node (Node): The node that has not yet been expanded (action that hasn't tried)
        """
        return NotImplemented

    @abstractmethod
    def evaluate(self, node: "Node"):
        """
        Simulate the game from expanded node (Rollout)

        Args:
            expanded_node (Node): The node that wants to get simulated

        Returns:
            value: 1 if win, -1 if lose, 0 if draw
        """
        return NotImplemented
    
    def backpropagate(self, node: "Node", reward):
        """
        Update the value of the node after simulation.
        
        Args:
            reward: Reward after the simulation step.
        """

        while node is not None:
            node.update(reward)
            node = node.parent

    def search(self):
        """
        Return best action in the current position
        """

        self._apply_root_noise()

        for _ in range(1, self.num_rollout + 1):
            selected_node = self.select()                        # Selection
            expanded_node = self.expand(selected_node)           # Expansion
            reward = self.evaluate(expanded_node)                # Simulation
            self.backpropagate(expanded_node, reward)            # Backpropagation
        
        return self.get_best_action()
    
    def get_child_visit_count(self) -> np.ndarray:
        """
        Returns the visit counts array in the current observed node -> np.ndarray with size 9
        """
        counts = np.zeros(9)
        for ch in self.observed.children:
            counts[ch.action] = ch._visit_count

        return counts
        
    def get_best_action(self):
        return int(np.argmax(self.get_child_visit_count()))
    
    def get_policy(self) -> list:
        """
        Returns the list of policy with the size 9
        for example [0.1, 0.2, 0, 0.15, 0, 0.05, 0.15, 0, 0.35]
        """
        counts = self.get_child_visit_count()
        total = counts.sum()
        if total == 0:
            raise ValueError("Children has no visit at all. Consider running search() first or if you have run it, increase the number of rollout (at least greater than 9).")

        return counts / total
    
    def advance(self, action, do_step=True):
        """
        Advance self.observed after performing action

        Returns:
            is_leaf, result
            is_leaf (bool): True if the board has winner or drawn, False otherwise
            result (None/int): None if the game still going, otherwise return the result (-1/1/0)
        """
        for child in self.observed.children:
            if child.action == action:
                if do_step and self.env is not None and isinstance(self.env, TicTacToe):
                    self.env.step(action)

                self.observed = child
                    
                assert np.all(self.observed.state == self.env.board)
                return self.observed.is_leaf, self.observed.result
            
        if action in TicTacToe.get_legal_actions(self.observed.state):
            self.observed.add_child_by_action(action)
            return self.advance(action, do_step)

        raise ValueError("No child with that action")

    @abstractmethod
    def _apply_root_noise(self):
        return NotImplemented
    
    def get_current_state(self):
        """
        Returns the current observed state
        """
        return self.observed.state.copy()
    
    def log(self, msg):
        if self.verbose:
            print(msg)
            

class VanillaMCTS(BaseMCTS):
    def __init__(self, env: TicTacToe = None, num_rollout=1000, exploration_constant=1.41):
        super().__init__(env, num_rollout)
        self.exploration_constant=exploration_constant


    def u(self, node: "Node"):
        if node._visit_count == 0:
            return float('inf')
        uct_value = node.q() + self.exploration_constant * (math.sqrt(np.log(node.parent._visit_count) / node._visit_count))
        return uct_value

    def expand(self, selected_node: "Node"):
        if selected_node.is_leaf:
            expanded_node = selected_node
        
        else:
            untried_action = selected_node.untried_actions.pop()
            turn = selected_node.turn
            child_state = TicTacToe.transition_state(selected_node.state, untried_action, turn)
            expanded_node = Node(child_state, selected_node, untried_action)

            selected_node.add_child(expanded_node)

        return expanded_node

    def evaluate(self, expanded_node: "Node"):
        if expanded_node.is_leaf:
            reward = expanded_node.result
        else:
            reward = TicTacToe.rollout(expanded_node.state)

        return reward

    def _apply_root_noise(self):
        return 


class NetworkMCTS(BaseMCTS):
    def __init__(self, network: PolicyValueNetwork, env: TicTacToe = None, num_rollout=1000, c_puct=1.41, epsilon=0.25, seed=42, alpha=0.03, add_noise=False, network_train=False):
        super().__init__(env, num_rollout, seed=seed)
        self.network = network
        self.network.train(network_train)
        self.c_puct = c_puct
        if add_noise:
            self.epsilon = epsilon
        self.alpha = alpha

    def u(self, node: "Node"):
        return self.c_puct * self.p(node) * math.sqrt(node.parent._visit_count) / (1 + node._visit_count)
    
    def p(self, node: "Node"):
        return (1 - self.epsilon) * node.prior + self.epsilon * node.noise

    def expand(self, selected_node: "Node"):
        if selected_node.is_leaf:
            return selected_node
        
        else:
            turn = selected_node.turn

            # For network MCTS, expand all first
            while not selected_node.is_fully_expanded():
                untried_action = selected_node.untried_actions.pop()
                expanded_state = TicTacToe.transition_state(selected_node.state, untried_action, turn)
                expanded_node = Node(expanded_state, selected_node, untried_action)

                selected_node.add_child(expanded_node)

        return selected_node

    def evaluate(self, selected_node: "Node"):
        """
        Updates the prior of the children of the selected node and evaluate the value of the node
        """
        if selected_node.is_leaf:
            return selected_node.result

        state = selected_node.state
        torch_state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0) 
        policy_head, value_head = self.network(torch_state)

        policy_head = policy_head.squeeze(0)
        value_head = value_head.item()
        
        # Updates the prior of the children
        children = selected_node.children
        legal_actions = TicTacToe.get_legal_actions(state)

        mask = np.ones(policy_head.shape, dtype=bool)
        mask[legal_actions] = False
        policy_head[mask] = 0

        policy_head /= torch.sum(policy_head)

        for child in children:
            a = child.action
            child.prior = policy_head[a].item()

        return value_head

    def _apply_root_noise(self):
        if self.epsilon == 0:
            return
        
        root = self.observed
        
        if not root.is_fully_expanded():
            self.expand(root) 
            self.evaluate(root) 
        
        legal_actions = TicTacToe.get_legal_actions(root.state)
        noise = self.rng.dirichlet([self.alpha] * len(legal_actions))
        
        action_to_noise = {a: noise[i] for i, a in enumerate(legal_actions)}
        
        for child in root.children:
            child.noise = action_to_noise[child.action]
    

class Node:
    """
    A node in the MCTS search tree.

    Each node holds a game state and the statistics gathered for it during search.

    Attributes:
        state: the game state this node represents.
        parent (Node | None): parent node, or None for the root.
        children (list[Node]): expanded child nodes.
        visits (int): number of times this node was visited.
        value (float): cumulative reward backpropagated through this node.
        uct_value (float): last computed UCT score (for selection).
    """

    def __init__(self, state=TicTacToe.base_state(), parent=None, action=None):
        """
        Create a node for state, optionally linked to parent.
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self._visit_count = 0
        self._value = 0.0
        self.prior = 0.0
        self.noise = 0.0
        self.turn = TicTacToe.get_whose_turn(self.state)
        self.is_leaf = TicTacToe.is_terminal(self.state)
        self.result = TicTacToe.get_result(self.state)
        self.untried_actions = TicTacToe.get_legal_actions(self.state)

    def add_child(self, child: "Node"):
        """
        Append child to this node's children.
        """
        self.children.append(child)

    def add_child_by_action(self, action: int):
        if action in TicTacToe.get_legal_actions(self.state):
            child_state = TicTacToe.transition_state(self.state, action)
            child_node = Node(child_state, self, action)
            self.add_child(child_node)
        else:
            raise ValueError("Action is illegal in the current state")
        return child_node

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def increment_visit(self):
        """
        Increment the n visit by one
        """
        self._visit_count += 1

    def update(self, value):
        """
        Record one visit and add value to the cumulative resward.
        Args:
            value: ABSOLUTE reward to backpropagate into this node. (-1 if X wins, 1 if O wins, 0 if draw)
         """
        self.increment_visit()
        value *= -self.turn      # Value is seen from parent's point of view
        self._value += value

    def q(self):
        """
        Return the mean value (value / visits), or 0 if never visited.

        Returns:
            float: average reward of this node.
        """

        if self._visit_count == 0:
            return 0
        return self._value / self._visit_count
                
    def __repr__(self):
        return f"Node(state=\n{self.state}, \nvisits={self._visit_count}, value={self._value})"
    
    def __copy__(self):
        """Implements copy.copy() behavior."""
        # Create a new instance, but keep original references for inner attributes
        return type(self)(self.state, self.parent)
    
    def copy(self):
        """Exposes a traditional .copy() method directly on the instance."""
        return copy.copy(self)
    


if __name__ == "__main__":
    network = PolicyValueNetwork()
    agent=NetworkMCTS(network=network, board=BASE_BOARD, num_rollout=1000)
    best_action = agent.search()


