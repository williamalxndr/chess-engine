import copy

from game.rules import Rules


class Node:
    """
    A node in the MCTS search tree.
    """

    def __init__(self, rules: Rules, state, parent=None, action=None):
        """
        Create a node for `state` and cache the rule-derived facts about it.

        Args:
            rules (Rules): game rules used to query this state.
            state: the game state this node represents.
            parent (Node): the node this one was expanded from, or None for the root.
            action (int): the action that led from `parent` to this node, or None.
        """
        self.rules = rules
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}             # dict action: node
        self._visit_count = 0
        self._value = 0.0
        self.priors = {}
        self.noise = {}
        self.turn = rules.get_whose_turn(state)
        self.is_leaf = rules.is_terminal(state)
        self.result = rules.get_result(state)
        self._legal_actions = rules.get_legal_actions(state) 
        self.untried_actions = list(self._legal_actions)  

    def add_child(self, child: "Node", action):
        """
        Attach `child` under the edge `action`.

        Args:
            child (Node): the child node to attach.
            action (int): the action edge that reaches `child`.
        """
        child.action = action
        self.children[action] = child

    def add_child_by_action(self, action: int):
        """
        Build and attach the child reached by `action`.

        Args:
            action (int): a legal action from this node's state.

        Returns:
            Node: the newly created child node.

        Raises:
            ValueError: if `action` is illegal in the current state.
        """
        if action not in self.get_legal_actions():
            raise ValueError("Action is illegal in the current state")

        child_state = self.rules.transition_state(self.state, action)
        child_node = Node(self.rules, child_state, self, action)
        self.add_child(child_node, action)
        return child_node

    def get_children_by_action(self, action) -> "Node":
        """
        Look up an existing child by its action.

        Args:
            action (int): the action edge to look up.

        Returns:
            Node: the child node, or None if it has not been created yet.
        """
        return self.children.get(action)

    def get_legal_actions(self):
        """
        Legal actions available from this node's state.

        Returns:
            list[int]: the legal actions.
        """
        return self._legal_actions

    def get_untried_action(self):
        """
        Pop and return one action that has not been expanded yet.

        Returns:
            int: an untried action (removed from untried_actions).
        """
        return self.untried_actions.pop()

    def get_leaf(self):
        """
        Whether this node is terminal (game over).

        Returns:
            bool: True if the state is terminal.
        """
        return self.is_leaf

    def get_result(self):
        """
        Game result at this node.

        Returns:
            int: -1 if X wins, 1 if O wins, 0 for a draw, or None if not terminal.
        """
        return self.result

    def is_fully_expanded(self):
        """
        Whether every legal action from this node has been expanded.

        Returns:
            bool: True if there are no untried actions left.
        """
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
    
    def apply_virtual_loss(self):
        node = self
        while node is not None:
            node._value -= 1
            node._visit_count += 1
            node = node.parent

    def remove_virtual_loss(self):
        node = self
        while node is not None:
            node._value += 1
            node._visit_count -= 1
            node = node.parent

    def __repr__(self):
        return f"Node(state=\n{self.state}, \nvisits={self._visit_count}, value={self._value})"

    def __copy__(self):
        """Implements copy.copy() behavior."""
        return type(self)(self.rules, self.state, self.parent)

    def copy(self):
        """Exposes a traditional .copy() method directly on the instance."""
        return copy.copy(self)
