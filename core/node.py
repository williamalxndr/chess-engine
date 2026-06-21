import copy

from game.rules import Rules


class Node:
    """
    A node in the MCTS search tree.

    Holds a game state and the statistics gathered for it during search.
    """

    def __init__(self, rules: Rules, state, parent=None, action=None, turn=None):
        self.rules = rules
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self._visit_count = 0
        self._value = 0.0
        self.prior = 0.0
        self.noise = 0.0
        self.turn = turn if turn is not None else (-1 if parent is None else -self.parent.turn)
        self.is_leaf = rules.is_terminal(state)
        self.result = rules.get_result(state)
        self.untried_actions = rules.get_legal_actions(state)

    def add_child(self, child: "Node"):
        """
        Append child to this node's children.
        """
        self.children.append(child)

    def add_child_by_action(self, action: int):
        if action not in self.rules.get_legal_actions(self.state):
            raise ValueError("Action is illegal in the current state")

        child_state = self.rules.transition_state(self.state, action)
        child_node = Node(self.rules, child_state, self, action)
        self.add_child(child_node)
        return child_node

    def get_leaf(self):
        return self.is_leaf

    def get_result(self):
        return self.result

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def increment_visit(self):
        """
        Increment the visit count by one.
        """
        self._visit_count += 1

    def update(self, value):
        """
        Record one visit and add value to the cumulative reward.

        Args:
            value: ABSOLUTE reward to backpropagate into this node
                (-1 if X wins, 1 if O wins, 0 if draw).
        """
        self.increment_visit()
        value *= -self.turn      # value is seen from the parent(opponent)'s point of view
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
        return type(self)(self.rules, self.state, self.parent)

    def copy(self):
        """Exposes a traditional .copy() method directly on the instance."""
        return copy.copy(self)



