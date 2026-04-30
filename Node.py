import math
import numpy as np

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children =[]
        self.visits = 0
        self.value = 0.0
        self.uct_value = 0.0
        self.children_by_state = {}

    def add_child(self, child: "Node"):
        self.children.append(child)
        self.children_by_state[child.state.tobytes()] = child
        
    def update(self, value):
        self.visits += 1
        self.value += value

    def get_value(self):
        if self.visits == 0:
            return 0
        return self.value / self.visits
    
    def uct(self, exploration_constant=1.41):
        if self.visits == 0:
            return float('inf')
        self.uct_value = self.get_value() + exploration_constant * (math.sqrt(np.log(self.parent.visits) / self.visits))
        return self.uct_value
    
    def best_child(self, exploration_constant=1.41):
        for child in self.children:
            child.uct_value = child.uct(exploration_constant)

        return max(self.children, key=lambda c: c.uct_value)
    
    def __repr__(self):
        return f"Node(state={self.state}, visits={self.visits}, value={self.value}, uct_value={self.uct_value})"
