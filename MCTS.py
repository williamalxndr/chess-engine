import sys

from env import TicTacToe
import numpy as np
import math
import copy

class MonteCarloTreeSearch:
    def __init__(self, env: TicTacToe, exploration_constant=1.41, epochs=1000):
        self.env = env
        self.exploration_constant = exploration_constant
        self.epochs = epochs
        self.root = Node(env.get_state())
        self.observed = self.root
        self.turn = env.turn

    def selection(self):
        """
        Select the best child, if no child are already expanded
        Returns:
            Node: the best child node to be expanded
        """
        curr = self.observed
        # print(curr)
        # print(f"untried actions: {curr.untried_actions}")
        # print(f"condition len(curr.untried_actions) == 0 = {len(curr.untried_actions) == 0}")
        # print(f"\n\nCHILDREN ({len(curr.children)}):")
        # print([ch.state for ch in curr.children])
        # print("\n\n")
        # print("=======================")

        # print(f"OLD CURR: {curr}")
        while len(curr.untried_actions) == 0:
            curr = curr.best_child()
        # print(f"NEW CURR: {curr}")

        return curr
    
    def expansion(self, selected_node: "Node"):
        """
        Expand the selected node with an untried action

        Args:
            selected_node (Node): The node that is selected by selection part

        Returns:
            expanded_node (Node): The node that has not yet been expanded (action that hasn't tried)
        """
        untried_action = selected_node.untried_actions.pop()
        turn = TicTacToe.get_whose_turn(selected_node.state)
        child_state = TicTacToe.transition_state(selected_node.state, untried_action, turn)
        child_node = Node(child_state, selected_node)

        selected_node.add_child(child_node)

        return child_node

    def simulation(self, expanded_node: "Node"):
        """
        Simulate the game from expanded node

        Args:
            expanded_node (Node): The node that wants to get simulated

        Returns:
            value: 1 if win, -1 if lose, 0 if draw
        """
        return TicTacToe.random_simulate(expanded_node.state)


    def backpropagation(self, expanded_node, reward):
        """
        Update the value of the node after simulation.
        
        Args:
            reward: Reward after the simulation step.
        """
        back = expanded_node
        while back is not None:
            back.update(reward)
            back = back.parent
        

    def run(self):
        for epoch in range(1, self.epochs + 1):
            selected_node = self.selection()                        # Selection
            expanded_node = self.expansion(selected_node)           # Expansion
            reward = self.simulation(expanded_node)                 # Simulation
            self.backpropagation(selected_node, reward)             # Backpropagation
            print(f"Epoch {epoch}: {self.root}")



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

    def __init__(self, state, parent=None):
        """
        Create a node for state, optionally linked to parent.
        """
        self.state = state
        self.parent = parent
        self.children = []
        self._n_visits = 0
        self._value = 0.0
        self._uct_value = self._uct()
        self.untried_actions()

    def add_child(self, child: "Node"):
        """
        Append child to this node's children.
        """
        self.children.append(child)

    def increment_visit(self):
        """
        Increment the n visit by one
        """
        self._n_visits += 1

    def update(self, value):
        """
        Record one visit and add value to the cumulative reward.
        Args:
            value: reward to backpropagate into this node.
         """
        self.increment_visit()
        self._value += value

    def untried_actions(self):
        """
        Update the untried action that is not already expanded. 
        """
        self.untried_actions = TicTacToe.get_legal_action(self.state)


    def _get_value(self):
        """
        Return the mean value (value / visits), or 0 if never visited.

        Returns:
            float: average reward of this node.
        """

        if self._n_visits == 0:
            return 0
        return self._value / self._n_visits
    
    def _uct(self, exploration_constant=1.41):
        """
        Compute this node's UCB1 score for selection.

        Balances exploitation (mean value) against exploration
        (parent visits vs. own visits). Unvisited nodes return
        infinity so they are always selected first.

        Args:
            exploration_constant: weight of the exploration term (default=sqrt(2)).

        Returns:
            float: the UCT score, or inf if this node has no visits.
        """

        if self._n_visits == 0:
            return float('inf')
        self._uct_value = self._get_value() + exploration_constant * (math.sqrt(np.log(self.parent._n_visits) / self._n_visits))
        return self._uct_value
    
    def best_child(self, exploration_constant=1.41):
        """Return the child with the highest UCT score.

        Args:
            exploration_constant: passed through to each child's uct().

        Returns:
            Node: the most promising child to descend into.
        """
        for child in self.children:
            child._uct_value = child._uct(exploration_constant)

        return max(self.children, key=lambda c: c._uct_value)
    
    def __repr__(self):
        return f"Node(state={self.state}, visits={self._n_visits}, value={self._value}, uct_value={self._uct_value})"
    
    def __copy__(self):
        """Implements copy.copy() behavior."""
        # Create a new instance, but keep original references for inner attributes
        return type(self)(self.state, self.parent)
    
    def copy(self):
        """Exposes a traditional .copy() method directly on the instance."""
        return copy.copy(self)




if __name__ == "__main__":
    env = TicTacToe()
    agent=MonteCarloTreeSearch(env, epochs=1000)
    agent.run()

    for child in agent.root.children:
        print(f"State: {child.state}, value: {child._value}, n_visit: {child._n_visits}, uct: {child._uct_value}")
