import threading
from abc import ABC, abstractmethod

import chess

from api import config
from core.encoder import int_to_move
from core.tree import NetworkMCTS
from game.rules import Rules


class GameService(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_bot_move(self, fen=None, moves=None):
        """
        ## Args:
            fen(str): position to move in, or None for the starting position
            moves(list[str]): UCI moves to play from `fen` before searching

        ## Returns:
            dict of move(str), fen(str), game_over(bool), result(int)
        """
        return NotImplemented

    @abstractmethod
    def evaluate_position(self, fen=None, moves=None):
        """
        Score a position without playing it. Analysis takes no side: the value
        is absolute, on the same axis as `result`.

        ## Args:
            fen(str): position to score, or None for the starting position
            moves(list[str]): UCI moves to play from `fen` before searching

        ## Returns:
            dict of value(float), fen(str), lines(list), game_over(bool), result(int)
        """
        return NotImplemented


class ChessService(GameService):
    def __init__(
        self,
        version=config.CHECKPOINT_VERSION,
        file_name=config.CHECKPOINT_FILE_NAME,
        parent_dir=config.CHECKPOINT_PARENT_DIR,
        num_rollout=config.NUM_ROLLOUT,
    ):
        super().__init__()
        self.rules = Rules.get("chess")
        self.bot = NetworkMCTS(
            game="chess",
            version=version,
            file_name=file_name,
            parent_dir=parent_dir,
            num_rollout=num_rollout,
            batch_size=config.MCTS_BATCH_SIZE,
            seed=config.SEED,
            add_noise=False,
            verbose=False,
        )
        # One search tree is shared by every request, so searches must not overlap.
        self._lock = threading.Lock()

    def build_board(self, fen=None, moves=None):
        """
        Rebuild the position the way UCI does: start from `fen`, then replay
        `moves`. Replaying keeps the move stack, which the rules need for
        threefold-repetition draws.

        Raises:
            ValueError: if the FEN or any move is invalid or illegal.
        """
        board = chess.Board() if fen is None else chess.Board(fen)

        for uci in moves or []:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise ValueError(f"illegal move '{uci}' in position '{board.fen()}'")
            board.push(move)

        return board

    def check_terminal(self, board):
        return self.rules.is_terminal(board)

    def get_bot_move(self, fen=None, moves=None):
        board = self.build_board(fen, moves)

        if self.check_terminal(board):
            return {
                "move": None,
                "fen": board.fen(),
                "game_over": True,
                "result": self.rules.get_result(board),
            }

        with self._lock:
            self.bot.reset(board.copy())
            action = self.bot.search()

        if action not in self.rules.get_legal_actions(board):
            raise RuntimeError(f"search returned illegal action {action}")

        move = int_to_move(action, board)
        board.push(move)

        return {
            "move": move.uci(),
            "fen": board.fen(),
            "game_over": self.check_terminal(board),
            "result": self.rules.get_result(board),
        }

    @staticmethod
    def absolute_value(node):
        """
        Node values are stored from the parent's point of view (see Node.update),
        so flipping by the side to move puts every node on the same axis as
        `result`: negative favours White, positive favours Black. Analysis takes
        no side, so this is the form the API reports.
        """
        return float(-node.turn * node.q())

    def principal_variation(self, child, length):
        """
        Follow the most-visited child down the tree, decoding each edge against
        the position its parent held.
        """
        pv = []
        node = child

        while len(pv) < length:
            pv.append(int_to_move(node.action, node.parent.state).uci())

            visited = [c for c in node.children.values() if c._visit_count > 0]
            if not visited:
                break
            node = max(visited, key=lambda c: c._visit_count)

        return pv

    def evaluate_position(self, fen=None, moves=None):
        board = self.build_board(fen, moves)

        if self.check_terminal(board):
            result = self.rules.get_result(board)
            return {
                "value": float(result),
                "fen": board.fen(),
                "lines": [],
                "game_over": True,
                "result": result,
            }

        with self._lock:
            self.bot.reset(board.copy())
            self.bot.search()

            root = self.bot.root
            value = self.absolute_value(root)

            candidates = [c for c in root.children.values() if c._visit_count > 0]
            candidates.sort(key=lambda c: c._visit_count, reverse=True)

            lines = [
                {
                    "move": int_to_move(child.action, root.state).uci(),
                    "value": self.absolute_value(child),
                    "visits": child._visit_count,
                    "pv": self.principal_variation(child, config.EVAL_PV_LENGTH),
                }
                for child in candidates[: config.EVAL_TOP_K]
            ]

        return {
            "value": value,
            "fen": board.fen(),
            "lines": lines,
            "game_over": False,
            "result": None,
        }


class GameServiceFactory:
    @staticmethod
    def make(game: str) -> GameService:
        if game.lower() == "chess":
            return ChessService()
        raise ValueError(f"unlisted game '{game}'")
