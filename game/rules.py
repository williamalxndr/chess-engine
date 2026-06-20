import numpy as np
from abc import ABC, abstractmethod


class IllegalBoardError(Exception):
    def __init__(self, message="Illegal board"):
        self.message = message
        super().__init__(self.message)


class Rules(ABC):
    """
    Pure game mechanics for a given game type.

    Every method takes an explicit `state` and never reads or mutates any
    instance attribute, a Rules object holds no game-in-progress data of
    its own. This is what lets one Rules instance be shared safely between
    a live Environment session and a search algorithm (e.g. MCTS) that needs
    to evaluate many candidate states at once, with zero ambiguity about
    which state is being checked.
    """

    @property
    @abstractmethod
    def action_space_size(self):
        """
        Returns:
            int: total number of possible actions for this game 
        """
        pass

    @abstractmethod
    def base_state(self):
        """
        Returns:
            the empty/initial state for this game
        """
        pass

    @abstractmethod
    def get_legal_actions(self, state):
        """
        Args:
            state: game state to check

        Returns:
            list: legal actions
        """
        pass

    @abstractmethod
    def is_terminal(self, state):
        """
        Args:
            state: game state to check

        Returns:
            bool: True if the game is over
        """
        pass

    @abstractmethod
    def get_result(self, state):
        """
        Args:
            state: terminal game state

        Returns:
            result value (e.g. -1, 0, 1), or None if not terminal
        """
        pass

    @abstractmethod
    def get_whose_turn(self, state):
        """
        Args:
            state: game state to check

        Returns:
            player identifier
        """
        pass

    @abstractmethod
    def transition_state(self, state, action, player=None):
        """
        Args:
            state: game state to apply the action to
            action: action to apply
            player: player making the move (optional, inferred if None)

        Returns:
            new game state
        """
        pass

    def rollout(self, state):
        """
        Default uniform-random rollout from `state` to a terminal state.

        Returns:
            result of the simulated game
        """
        state = state.copy()
        while not self.is_terminal(state):
            legal = self.get_legal_actions(state)
            action = np.random.choice(legal)
            state = self.transition_state(state, action)
        return self.get_result(state)


class TicTacToeRules(Rules):
    """
    3x3 Tic-Tac-Toe rules. Board convention: -1 = X, 1 = O, 0 = empty.
    X always moves first.
    """

    @property
    def action_space_size(self):
        return 9

    def base_state(self):
        return np.zeros((3, 3), dtype=int)

    def get_whose_turn(self, state):
        unique, counts = np.unique(state, return_counts=True)
        counts_by_player = dict(zip(unique, counts))
        x = counts_by_player.get(-1, 0)
        o = counts_by_player.get(1, 0)

        if o == x:
            return -1       # X's turn
        elif o + 1 == x:
            return 1        # O's turn
        raise IllegalBoardError("Board is illegal: inconsistent turn counts")

    def get_legal_actions(self, state):
        return [i for i in range(9) if state[divmod(i, 3)] == 0]

    def transition_state(self, state, action, player=None):
        if player is None:
            player = self.get_whose_turn(state)

        row, col = divmod(action, 3)
        if state[row, col] != 0:
            raise ValueError("That cell is already occupied")

        new_state = state.copy()
        new_state[row, col] = player
        return new_state

    def check_winner(self, state, player):
        for i in range(3):
            if np.all(state[i, :] == player) or np.all(state[:, i] == player):
                return True
        if np.all(np.diag(state) == player) or np.all(np.diag(np.fliplr(state)) == player):
            return True
        return False

    def check_draw(self, state):
        return np.all(state != 0)

    def is_terminal(self, state):
        return (
            self.check_winner(state, -1)
            or self.check_winner(state, 1)
            or self.check_draw(state)
        )

    def get_result(self, state):
        if self.check_winner(state, -1):
            return -1
        if self.check_winner(state, 1):
            return 1
        if self.check_draw(state):
            return 0
        return None

    def is_legal_board(self, state):
        if not (isinstance(state, np.ndarray) and state.shape == (3, 3)):
            raise IllegalBoardError("Board has wrong type or shape")

        unique_vals, counts = np.unique(state, return_counts=True)
        freq = dict(zip(unique_vals, counts))
        x, o, empty = freq.get(-1, 0), freq.get(1, 0), freq.get(0, 0)

        if (empty + x + o) == 9 and (x == o or x - 1 == o):
            return True

        raise IllegalBoardError("Board has inconsistent piece counts")
    


RULES_REGISTRY = {
    "tictactoe": TicTacToeRules(),
    "chess": NotImplemented
}