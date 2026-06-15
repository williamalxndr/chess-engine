import gym
import numpy as np

# Environment for the game

class TicTacToe(gym.Env):
    def __init__(self):
        """
        Initialize the environment with an empty board.
        X always moves first.

        Attributes:
            board (np.ndarray): 3x3 game state (-1: X, 1: O, 0 = empty).
            observation_space (gym.spaces.Box): board value range.
            action_space (gym.spaces.Discrete): 9 possible cell indices.
            turn (int): player allowed to move next (-1 or 1).
        """
        self.board = np.zeros((3, 3), dtype=int)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3, 3), dtype=int)
        self.action_space = gym.spaces.Discrete(9)
        self.reset()

    def reset(self, board=None):
        """
        Reset the board to empty and choose the starting player.

        Args:
            starting_player: player who moves first 
                If None, chosen randomly.

        Returns:
            np.ndarray: a copy of the empty board.
        """
        if board is None:
            self.board = np.zeros((3, 3), dtype=int)
        else:
            self.board = board

        self.turn = TicTacToe.get_whose_turn(self.board)
 
        return self.board
        
    
    def step(self, action):
        """
        Make a move for player.
        Args: 
            action: board index to be filled by player

        Returns:
            observation (np.ndarray): game state after player makes a move
            reward (int): -1 if X win, 1 if O wins, O if draw or the game continues
            terminated (bool): True if the game is over, False if the game still continues
            info (dict): "winner" (player / 0 /None) 

        Raises:
            ValueError: if it is not player's turn, or the action is illegal
        """
        current_player = TicTacToe.get_whose_turn(self.board)

        # Action validation
        if action not in self.get_legal_actions_self():
            raise ValueError("Action is illegal!")
        
        self.board = self.transition_state(self.board, action, current_player)  # Player's move (fill the empty cell with player's mark)

        if self.is_terminal_self():
            terminal = True
            reward = self.get_result_self()
        else:
            terminal = False
            reward = 0
        
        self.turn *= -1  # Switch player
        return self.board, reward, terminal, {"winner": None}  # Game continues
    
    def get_state(self):
        return self.board
    
    def check_winner_self(self, player):
        """
        Check whether player is winning

        Args:
            player: player to check (-1 or 1)
        Returns:
            bool: True if player occupies a full row, column or diagonal

        """
        return self.check_winner(self.board, player)
    
    def check_draw_self(self):
        """
        Return True if the board is full. Assumes no winner has been found.

        Returns:
            bool: True if draw, False if the game still running
        """
        return self.check_draw(self.state)
    
    def get_legal_actions_self(self):
        """
        Returns indices of empty cells (the legal moves).
        
        Returns:
            list[int]: cell indices 0-8 that are unoccupied
        """
        return self.get_legal_action(self.board)
    
    def render(self):
        """
        Prints the board
        """
        symbols = {0: ' ', 1: 'O', -1: 'X'}
        for row in self.board:
            print(' | '.join(symbols[cell] for cell in row))
            print('-' * 9)

    
    @staticmethod
    def rollout(board):
        """
        Randomly simulate a game for board

        Args:
            board(np.ndarray): The board to be simulated/rollout

        Returns:
            player(int): -1 if X wins, 0 if draw, 1 if O wins 
        """
        board_copy = board.copy()
        player = TicTacToe.get_whose_turn(board_copy)
        
        # Check if already terminates:
        if TicTacToe.check_winner(board_copy, player):
            return player

        legal_action = TicTacToe.get_legal_action(board_copy)
        random_action = np.random.choice(legal_action)

        terminate = False

        while not terminate:
            board_copy = TicTacToe.transition_state(board_copy, random_action, player)
            win = TicTacToe.check_winner(board_copy, player)
            draw = TicTacToe.check_draw(board_copy)
            terminate = win or draw

            random_action_idx = np.argwhere(legal_action==random_action)
            legal_action = np.delete(legal_action, random_action_idx)
            if not terminate:
                player *= -1
                random_action = np.random.choice(legal_action)



        reward = 1 if win else 0

        return player * reward
        

    @staticmethod
    def get_whose_turn(board):
        unique, counts = np.unique(board, return_counts=True)
        dict_unique = dict(zip(unique, counts))
        x = dict_unique.get(-1, 0)
        o = dict_unique.get(1, 0)

        if o == x:
            return -1       # X turn, (1)
        elif o+1 == x:
            return 1        # O turn, (-1)
        else:
            raise ValueError("Board is illegal!")
        
    @staticmethod
    def get_legal_action(board):
        return [i for i in range(9) if board[divmod(i, 3)] == 0]

    @staticmethod
    def transition_state(board: np.ndarray, action, turn=None):
        if turn is None:
            turn = TicTacToe.get_whose_turn(board)

        board_copy = board.copy()
        row, col = divmod(action, 3)

        if board_copy[row][col] == 0:
            board_copy[row][col] = turn
        else:
            raise ValueError("That cell is already occupied")

        return board_copy

    @staticmethod
    def check_winner(board, player):
        for i in range(3):
            if np.all(board[i, :] == player) or np.all(board[:, i] == player):
                return True
        if np.all(np.diag(board) == player) or np.all(np.diag(np.fliplr(board)) == player):
            return True
        return False


    @staticmethod
    def check_draw(board):
        return np.all(board)
    
    @staticmethod
    def is_terminal(board):
        draw = np.all(board)
        win = TicTacToe.check_winner(board, -1) or TicTacToe.check_winner(board, 1)
        return win or draw
    
    def is_terminal_self(self):
        return TicTacToe.is_terminal(self.board)
    
    @staticmethod
    def get_result(board):
        if TicTacToe.check_winner(board, -1):
            return -1
        elif TicTacToe.check_winner(board, 1):
            return 1
        elif TicTacToe.check_draw(board):
            return 0
        return None
    
    def get_result_self(self):
        return TicTacToe.get_result(self.board)
    
    @staticmethod
    def base_state():
        return np.zeros([3,3])
        


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
        state, reward, done, info = env.step(action)

        if "error" in info:
            print(info["error"])
            continue

        if reward == 1:
            env.render()
            print(f"Player {player_name[current_player]} wins!")
        elif reward == 0 and done:
            env.render()
            print("It's a draw!")

        current_player *= -1  # Switch player
