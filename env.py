import gym
import numpy as np

# Environment for the game

class TicTacToe(gym.Env):
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3, 3), dtype=int)
        self.action_space = gym.spaces.Discrete(9)
        self.turn = np.random.choice([1, -1])  # Randomly choose which player starts

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.turn = np.random.choice([1, -1])  # Randomly choose which player starts
        return self.board.copy()
    
    def step(self, action, player):
        self.turn = player

        row, col = divmod(action, 3)

        if action not in self._legal_actions():
            return self.board, -1, False, {"error": "Invalid action! Choose an empty cell from 0-8."}
        
        self.board[row, col] = player  # Player's move

        if self.check_winner(player):
            return self.board, player, True, {"winner": player}  # Player wins

        if self.check_draw():
            return self.board, 0, True, {"winner": 0}  # Draw
        
        self.turn *= -1  # Switch player
        return self.board, 0, False, {"winner": None}  # Game continues
    
    def check_winner(self, player):
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False
    
    def check_draw(self):
        return np.all(self.board != 0)
    
    def _legal_actions(self):
        return [i for i in range(9) if self.board[divmod(i, 3)] == 0]
    
    def render(self):
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        for row in self.board:
            print(' | '.join(symbols[cell] for cell in row))
            print('-' * 9)
    

# Helper function
def game_simulate(state, action, player):
    env = TicTacToe()
    env.board = state.copy()
    return env.step(action, player)

def check_win(board, player):
    for i in range(3):
        if np.all(board[i, :] == player) or np.all(board[:, i] == player):
            return True
    if np.all(np.diag(board) == player) or np.all(np.diag(np.fliplr(board)) == player):
        return True
    return False


if __name__ == "__main__":
    env = TicTacToe()
    state = env.reset()
    done = False
    current_player = np.random.choice([1, -1])  # Randomly choose which player starts

    player_name = {1: "X", -1: "O"}

    print("Welcome to Tic Tac Toe!")
    print(f"Player {player_name[current_player]} starts first.")

    while not done:
        env.render()
        action = int(input(f"Player {player_name[current_player]} turn. Enter action (0-8): "))
        state, reward, done, info = env.step(action, current_player)

        if "error" in info:
            print(info["error"])
            continue

        if reward == current_player:
            env.render()
            print(f"Player {player_name[current_player]} wins!")
        elif reward == 0 and done:
            env.render()
            print("It's a draw!")

        current_player *= -1  # Switch player
