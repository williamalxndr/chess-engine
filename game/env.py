import gym
from abc import ABC, abstractmethod
import chess

from game.rules import Rules, TicTacToeRules, ChessRules


class Environment(ABC):
    """
    A stateful game session. Owns `self.state` and the reset/step/render
    lifecycle; delegates all rule evaluation to a `Rules` instance instead
    of implementing game mechanics itself. Session management and game
    mechanics are two separate responsibilities that can each change
    independently of the other.
    """

    def __init__(self, rules: Rules):
        self.rules = rules

    def reset(self, state=None):
        """
        Reset environment to initial state.

        Returns:
            state: initial observation
        """
        self.state = self.rules.base_state() if state is None else state
        return self.state

    def step(self, action):
        """
        Apply action to the live environment state.

        Args:
            action: the action to take

        Returns:
            observation: next state
            reward: reward signal
            terminated: whether the episode is done
            info: dict with extra info
        """
        if action not in self.rules.get_legal_actions(self.state):
            raise ValueError("Action is illegal!")

        self.state = self.rules.transition_state(self.state, action)

        terminated = self.rules.is_terminal(self.state)
        result = self.rules.get_result(self.state) if terminated else None
        reward = result if terminated else 0

        return self.state, reward, terminated, {"winner": result}
    
    def get_legal_actions(self):
        return self.rules.get_legal_actions(self.state)

    @abstractmethod
    def render(self):
        """
        Render current state to stdout or display.
        """
        pass


class TicTacToe(Environment):
    def __init__(self):
        super().__init__(rules=TicTacToeRules())
        self.reset()

    def render(self):
        """
        Prints the live board.
        """
        symbols = {0: ' ', 1: 'O', -1: 'X'}
        for row in self.state:
            print(' | '.join(symbols[cell] for cell in row))
            print('-' * 9)


class Chess(Environment, gym.Env):
    def __init__(self):
        super().__init__(rules=ChessRules())
        self.reset()

    def render(self):
        return NotImplemented


if __name__ == "__main__":
    env = TicTacToe()
    state = env.reset()
    done = False

    player_name = {-1: "X", 1: "O"}

    print("Welcome to Tic Tac Toe!")

    while not done:
        env.render()
        current_player = env.rules.get_whose_turn(env.state)
        action = int(input(f"Player {player_name[current_player]} turn. Enter action (0-8): "))
        state, reward, done, info = env.step(action)

        if done:
            env.render()
            if reward == 0:
                print("It's a draw!")
            else:
                print(f"Player {player_name[reward]} wins!")