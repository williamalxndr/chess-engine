import sys

from Node import Node
from env import game_simulate, TicTacToe
import numpy as np
import math

class MCTS:
    def __init__(self, env=None, num_simulations=1000, verbose=False, initial_state=None, root_player=1):
        if env is None:
            env = TicTacToe()
        self.env = env
        self.num_simulations = num_simulations
        self.verbose = verbose
        if initial_state is None:
            initial_state = self.env.reset()
        self.env.board = initial_state.copy()
        self.env.turn = root_player
        self.node = Node(state=initial_state.copy())
        self.observed_node = self.node  # Start with the root node as the observed node
        self.root_player = root_player

    def player_name(self, player):
        return {1: "X", -1: "O", 0: "Draw", None: "None"}[player]

    def show_board(self, state):
        symbols = {0: " ", 1: "X", -1: "O"}
        print("  0   1   2")
        for row_index, row in enumerate(state):
            cells = " | ".join(symbols[cell] for cell in row)
            print(f"{row_index} {cells}")
            if row_index < 2:
                print("  ---------")

    def log(self, message):
        if self.verbose:
            print(message)

    def legal_actions(self, state):
        return [i for i in range(9) if state[divmod(i, 3)] == 0]

    def find_child_for_state(self, node, state):
        for child in node.children:
            if np.array_equal(child.state, state):
                return child
        return None

    def advance_to_state(self, state):
        next_node = self.find_child_for_state(self.observed_node, state)
        if next_node is None:
            next_node = Node(state=state.copy())
            self.log("No matching child existed. Created a new node for the current board.")
        else:
            self.log("Reused an existing expanded node for the current board.")

        next_node.parent = None
        self.node = next_node
        self.observed_node = next_node

    def winner(self, state):
        for player in (1, -1):
            for i in range(3):
                if np.all(state[i, :] == player) or np.all(state[:, i] == player):
                    return player
            if np.all(np.diag(state) == player) or np.all(np.diag(np.fliplr(state)) == player):
                return player
        if not self.legal_actions(state):
            return 0
        return None

    def player_to_move(self, state):
        move_count = np.count_nonzero(state != 0)
        return self.root_player if move_count % 2 == 0 else -self.root_player

    def selection(self, node):
        self.log("\nSelection")
        self.log(f"Current node: {node}")
        if self.verbose:
            self.show_board(node.state)

        if self.winner(node.state) is not None:
            self.log("Selected node is terminal.")
            return node

        player = self.player_to_move(node.state)

        # If not all children are expanded, expand one of the unexpanded children
        # If all children are expanded, select the child with the highest UCT value
        for action in self.legal_actions(node.state):
            next_state, _, _, _ = game_simulate(node.state, action, player)
            if self.find_child_for_state(node, next_state) is None:
                self.log(f"Found unexpanded action {action}.")
                return node
            
        # If all children are expanded, select the child with the highest UCT value, then select the unexpanded child of that node
        best_child = node.best_child()
        self.log("All actions expanded. Moving to best UCT child.")
        self.log(f"Best UCT child: {best_child}")
        return self.selection(best_child) 
    
    def expansion(self, node):
        self.log("\nExpansion")
        self.log(f"Expanding from node: {node}")
        if self.winner(node.state) is not None:
            self.log("Node is terminal. Skipping expansion.")
            return node

        player = self.player_to_move(node.state)
        for action in self.legal_actions(node.state):
            next_state, _, _, _ = game_simulate(node.state, action, player=player)
            if self.find_child_for_state(node, next_state) is None:
                child = Node(state=next_state, parent=node)
                node.add_child(child)
                self.log(f"Expanded action {action} for player {self.player_name(player)}.")
                self.log(f"New child node: {child}")
                if self.verbose:
                    self.show_board(child.state)
                return child

        best_child = node.best_child()
        self.log(f"No unexpanded actions. Reusing best child: {best_child}")
        return best_child

    def simulation(self, state):
        # Simulate a random game from the given state and return the reward
        self.log("\nSimulation")
        current_state = state.copy()
        current_player = self.player_to_move(current_state)
        step_number = 1

        while True:
            winner = self.winner(current_state)
            if winner is not None:
                self.log(f"Rollout winner: {self.player_name(winner)}.")
                return winner

            legal_actions = self.legal_actions(current_state)
            if not legal_actions:
                self.log("Rollout ended in a draw.")
                return 0  # Draw
            action = np.random.choice(legal_actions)
            current_state, _, done, info = game_simulate(current_state, action, current_player)
            self.log(f"Rollout step {step_number}: player {self.player_name(current_player)} chose action {action}.")
            if self.verbose:
                self.show_board(current_state)
            if done:
                self.log(f"Rollout winner: {self.player_name(info['winner'])}.")
                return info["winner"]  # Return the winner
            current_player *= -1  # Switch player
            step_number += 1

    def backpropagation(self, node, reward):
        # Update the node's value and visit count, then propagate the reward up to the parent nodes
        self.log("\nBackpropagation")
        self.log(f"Reward: {self.player_name(reward)}")
        while node is not None:
            self.log(f"Before update: {node}")
            before_visits = node.visits
            before_value = node.value
            node.update(reward)
            self.log(
                f"Node updated: visits {before_visits} -> {node.visits}, "
                f"value {before_value} -> {node.value}"
            )
            self.log(f"After update: {node}")
            node = node.parent

    def best_action(self):
        best_child = max(self.observed_node.children, key=lambda c: c.visits)  # Select the child with the most visits
        player = self.player_to_move(self.observed_node.state)
        for action in self.legal_actions(self.observed_node.state):
            next_state, _, _, _ = game_simulate(self.observed_node.state, action, player=player)
            if np.array_equal(best_child.state, next_state):
                return action
        return None  
    

    def run(self):
        self.log("Root board")
        if self.verbose:
            self.show_board(self.observed_node.state)
            self.log(f"Root player: {self.player_name(self.root_player)}")

        for simulation_number in range(1, self.num_simulations + 1):
            self.log(f"\n========== Simulation {simulation_number} ==========")
            select_node = self.selection(self.observed_node)  # Selection
            expanded_node = self.expansion(select_node)  # Expansion
            reward = self.simulation(expanded_node.state)  # Simulation
            self.backpropagation(expanded_node, reward)  # Backpropagation


def choose_mcts_action(board, num_simulations=1000, verbose=False):
    mcts = MCTS(
        num_simulations=num_simulations,
        verbose=verbose,
        initial_state=board,
        root_player=1,
    )
    mcts.run()
    action = mcts.best_action()
    if action is None:
        legal_actions = mcts.legal_actions(board)
        return np.random.choice(legal_actions) if legal_actions else None
    return action


def play_against_mcts(num_simulations=200):
    env = TicTacToe()
    initial_state = env.reset()
    current_player = choose_first_player()
    mcts = MCTS(
        num_simulations=num_simulations,
        verbose=False,
        initial_state=initial_state,
        root_player=current_player,
    )
    done = False
    player_name = {1: "X", -1: "O", 0: "Draw"}

    print("You are O. MCTS is X.")
    print("Actions are 0-8, left to right, top to bottom.")
    print(f"{'MCTS' if current_player == 1 else 'You'} move first.")

    while not done:
        env.render()

        if current_player == 1:
            print("Bot is thinking...")
            mcts.num_simulations = num_simulations
            mcts.run()
            action = mcts.best_action()
            print(f"Bot chose action {action}.")
        else:
            try:
                action = int(input("Your turn (O). Enter action (0-8): "))
            except ValueError:
                print("Invalid input! Enter a number from 0-8.")
                continue
            except EOFError:
                print()
                break

        state, reward, done, info = env.step(action, current_player)

        if "error" in info:
            print(info["error"])
            continue

        mcts.advance_to_state(state)
        print(f"Bot tree root visits: {mcts.observed_node.visits}, children: {len(mcts.observed_node.children)}")

        if done:
            env.render()
            winner = info["winner"]
            if winner == 0:
                print("It's a draw!")
            else:
                print(f"Player {player_name[winner]} wins!")
            break

        current_player *= -1


def choose_first_player():
    while True:
        choice = input("Who moves first? Enter 'me' for you (O), or 'bot' for Bot (X): ").strip().lower()
        if choice in ("me", "you", "player", "human", "o"):
            return -1
        if choice in ("bot", "mcts", "x"):
            return 1
        print("Invalid choice. Type 'me' or 'bot'.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "play":
        play_against_mcts()
    else:
        mcts = MCTS(num_simulations=5, verbose=True)
        mcts.run()
        best_action = mcts.best_action()
        print(f"Best action: {best_action}")

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children =[]
        self.visits = 0
        self.value = 0.0
        self.uct_value = 0.0

    def add_child(self, child: "Node"):
        self.children.append(child)
        
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