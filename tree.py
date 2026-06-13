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

    def select(self):
        """
        Select the best child, if no child are already expanded
        Returns:
            Node: the best child node to be expanded
        """
        selected_node = self.observed
        while selected_node.is_fully_expanded() and not selected_node.leaf:
            selected_node = selected_node.best_child()
        
        return selected_node
    
    def expand(self, selected_node: "Node"):
        """
        Expand the selected node with an untried action

        Args:
            selected_node (Node): The node that is selected by selection part

        Returns:
            expanded_node (Node): The node that has not yet been expanded (action that hasn't tried)
        """
        if selected_node.leaf:
            expanded_node = selected_node
        
        else:
            untried_action = selected_node.untried_actions.pop()
            turn = selected_node.turn
            child_state = TicTacToe.transition_state(selected_node.state, untried_action, turn)
            expanded_node = Node(child_state, selected_node, untried_action)

            selected_node.add_child(expanded_node)

        return expanded_node

    def rollout(self, expanded_node: "Node"):
        """
        Simulate the game from expanded node

        Args:
            expanded_node (Node): The node that wants to get simulated

        Returns:
            value: 1 if win, -1 if lose, 0 if draw
        """
        if expanded_node.leaf:
            reward = expanded_node.winner
        else:
            reward = TicTacToe.rollout(expanded_node.state)

        return reward

    def backpropagate(self, expanded_node, reward):
        """
        Update the value of the node after simulation.
        
        Args:
            reward: Reward after the simulation step.
        """
        back = expanded_node
        

        while back is not None:
            back.update(reward)
            back = back.parent
        
    def search(self):
        """
        Run the MCTS algorithm
        Returns:
            action(int): the best action after running the MCTS algorithm
        """
        for epoch in range(1, self.epochs + 1):
            selected_node = self.select()                        # Selection
            expanded_node = self.expand(selected_node)           # Expansion
            reward = self.rollout(expanded_node)                 # Simulation
            self.backpropagate(expanded_node, reward)             # Backpropagation
        
        return self.get_best_action()

        
    def get_best_action(self):
        best_child = max(self.observed.children, key=lambda child: child._value/child._n_visits)  # Greedy, not using UCT value
        best_action = best_child.action
        return best_action

    def advance(self, action):
        for child in self.observed.children:
            if child.action == action:
                self.observed = child
                return
        raise ValueError("No child with that action")
    

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

    def __init__(self, state, parent=None, action=None):
        """
        Create a node for state, optionally linked to parent.
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self._n_visits = 0
        self._value = 0.0
        self._uct_value = self.uct_score()
        self.turn = TicTacToe.get_whose_turn(self.state)
        self.leaf = TicTacToe.is_terminal(self.state)
        if self.leaf:
            self.winner = TicTacToe.get_winner(self.state)
        self.untried_actions = TicTacToe.get_legal_action(self.state)

    def add_child(self, child: "Node"):
        """
        Append child to this node's children.
        """
        self.children.append(child)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def increment_visit(self):
        """
        Increment the n visit by one
        """
        self._n_visits += 1

    def update(self, value):
        """
        Record one visit and add value to the cumulative resward.
        Args:
            value: reward to backpropagate into this node. (-1 if X wins, 1 if O wins, 0 if draw)
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

        if self._n_visits == 0:
            return 0
        return self._value / self._n_visits
    
    def uct_score(self, exploration_constant=1.41):
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
        self._uct_value = self.q() + exploration_constant * (math.sqrt(np.log(self.parent._n_visits) / self._n_visits))
        return self._uct_value
    
    def best_child(self, exploration_constant=1.41):
        """Return the child with the highest UCT score.

        Args:
            exploration_constant: passed through to each child's uct().

        Returns:
            Node: the most promising child to descend into.
        """
        for child in self.children:
            child._uct_value = child.uct_score(exploration_constant)

        return max(self.children, key=lambda c: c._uct_value)
        
    def __repr__(self):
        return f"Node(state=\n{self.state}, \nvisits={self._n_visits}, value={self._value}, uct_value={self._uct_value})"
    
    def __copy__(self):
        """Implements copy.copy() behavior."""
        # Create a new instance, but keep original references for inner attributes
        return type(self)(self.state, self.parent)
    
    def copy(self):
        """Exposes a traditional .copy() method directly on the instance."""
        return copy.copy(self)
    


if __name__ == "__main__":
    env = TicTacToe()
    agent=MonteCarloTreeSearch(env, epochs=10)
    best_action = agent.search()
    print(best_action)


