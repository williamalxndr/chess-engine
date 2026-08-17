import unittest
from unittest.mock import patch

import chess

from api.app import app
from api.service import ChessService

START_FEN = chess.Board().fen()
# White queens on f7/g7: black is checkmated, so the position is already terminal.
CHECKMATE_FEN = "7k/5QQ1/8/8/8/8/8/7K b - - 0 1"
# Black has no legal move but is not in check.
STALEMATE_FEN = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"


class FakeChessService:
    """Records what the route passed in, so route tests need no network."""

    def __init__(self, result=None, error=None, evaluation=None):
        self.calls = []
        self.error = error
        self.result = result or {
            "move": "e2e4",
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "game_over": False,
            "result": None,
        }
        self.evaluation = evaluation or {
            "value": -0.25,
            "fen": START_FEN,
            "lines": [{"move": "e2e4", "value": -0.3, "visits": 12, "pv": ["e2e4", "e7e5"]}],
            "game_over": False,
            "result": None,
        }

    def get_bot_move(self, fen=None, moves=None):
        self.calls.append({"fen": fen, "moves": moves})
        if self.error is not None:
            raise self.error
        return self.result

    def evaluate_position(self, fen=None, moves=None):
        self.calls.append({"fen": fen, "moves": moves})
        if self.error is not None:
            raise self.error
        return self.evaluation


class MoveRouteTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def post(self, body, game="chess"):
        return self.client.post(f"/api/games/{game}/move", json=body)

    def test_returns_bot_move_in_success_envelope(self):
        service = FakeChessService()
        with patch("api.routes.get_service", return_value=service):
            response = self.post({"fen": START_FEN, "moves": ["e2e4", "e7e5"]})

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["status"], 200)
        self.assertEqual(body["data"], service.result)
        self.assertEqual(
            service.calls, [{"fen": START_FEN, "moves": ["e2e4", "e7e5"]}]
        )

    def test_empty_body_falls_back_to_start_position(self):
        service = FakeChessService()
        with patch("api.routes.get_service", return_value=service):
            response = self.post({})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(service.calls, [{"fen": None, "moves": []}])

    def test_terminal_position_reports_no_move(self):
        service = FakeChessService(
            result={
                "move": None,
                "fen": CHECKMATE_FEN,
                "game_over": True,
                "result": -1,
            }
        )
        with patch("api.routes.get_service", return_value=service):
            response = self.post({"fen": CHECKMATE_FEN})

        data = response.get_json()["data"]
        self.assertIsNone(data["move"])
        self.assertTrue(data["game_over"])
        self.assertEqual(data["result"], -1)

    def test_unknown_game_returns_400(self):
        response = self.post({}, game="tictactoe")

        self.assertEqual(response.status_code, 400)
        self.assertIn("unrecognized game", response.get_json()["message"])

    def test_wrong_field_type_returns_400(self):
        response = self.post({"moves": "e2e4"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("moves", response.get_json()["message"])

    def test_service_value_error_returns_400(self):
        service = FakeChessService(error=ValueError("illegal move 'e2e5'"))
        with patch("api.routes.get_service", return_value=service):
            response = self.post({"moves": ["e2e5"]})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["message"], "illegal move 'e2e5'")


class EvaluateRouteTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def post(self, body, game="chess"):
        return self.client.post(f"/api/games/{game}/evaluate", json=body)

    def test_returns_evaluation_in_success_envelope(self):
        service = FakeChessService()
        with patch("api.routes.get_service", return_value=service):
            response = self.post({"fen": START_FEN, "moves": []})

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["status"], 200)
        self.assertEqual(body["data"], service.evaluation)

    def test_empty_body_falls_back_to_start_position(self):
        service = FakeChessService()
        with patch("api.routes.get_service", return_value=service):
            response = self.post({})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(service.calls, [{"fen": None, "moves": []}])

    def test_unknown_game_returns_400(self):
        response = self.post({}, game="tictactoe")

        self.assertEqual(response.status_code, 400)
        self.assertIn("unrecognized game", response.get_json()["message"])

    def test_wrong_field_type_returns_400(self):
        response = self.post({"moves": "e2e4"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("moves", response.get_json()["message"])

    def test_service_value_error_returns_400(self):
        service = FakeChessService(error=ValueError("illegal move 'e2e5'"))
        with patch("api.routes.get_service", return_value=service):
            response = self.post({"moves": ["e2e5"]})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["message"], "illegal move 'e2e5'")


class StaticFrontendTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_root_serves_the_frontend_shell(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<title>Chess Engine</title>", response.data)

    def test_frontend_assets_are_served_from_the_same_origin(self):
        for path in ("/styles.css", "/src/app.js", "/src/api.js", "/src/chess.js"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)


class ChessServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # file_name=None builds a fresh network instead of loading a checkpoint,
        # so these tests do not depend on gitignored weights.
        cls.service = ChessService(file_name=None, num_rollout=8)

    def test_network_is_served_in_eval_mode(self):
        # Training mode would let BatchNorm normalise each MCTS batch by its own
        # statistics, so a position's score would depend on the other leaves it
        # happened to be batched with.
        self.assertFalse(self.service.bot.network.training)

    def test_build_board_defaults_to_start_position(self):
        board = self.service.build_board()

        self.assertEqual(board.fen(), START_FEN)

    def test_build_board_replays_moves_and_keeps_history(self):
        board = self.service.build_board(moves=["e2e4", "e7e5"])

        self.assertEqual(board.move_stack, [chess.Move.from_uci("e2e4"),
                                            chess.Move.from_uci("e7e5")])
        self.assertEqual(board.turn, chess.WHITE)

    def test_build_board_rejects_unparsable_fen(self):
        with self.assertRaises(ValueError):
            self.service.build_board(fen="not a fen")

    def test_build_board_rejects_illegal_move(self):
        with self.assertRaises(ValueError):
            self.service.build_board(moves=["e2e5"])

    def test_checkmate_returns_no_move_and_white_win(self):
        result = self.service.get_bot_move(fen=CHECKMATE_FEN)

        self.assertIsNone(result["move"])
        self.assertTrue(result["game_over"])
        self.assertEqual(result["result"], -1)
        self.assertEqual(result["fen"], CHECKMATE_FEN)

    def test_stalemate_returns_draw(self):
        result = self.service.get_bot_move(fen=STALEMATE_FEN)

        self.assertIsNone(result["move"])
        self.assertTrue(result["game_over"])
        self.assertEqual(result["result"], 0)

    def test_search_returns_a_legal_move_and_advances_the_position(self):
        board = chess.Board()
        result = self.service.get_bot_move()

        move = chess.Move.from_uci(result["move"])
        self.assertIn(move, board.legal_moves)

        board.push(move)
        self.assertEqual(result["fen"], board.fen())
        self.assertFalse(result["game_over"])
        self.assertIsNone(result["result"])

    def test_evaluate_scores_the_start_position(self):
        evaluation = self.service.evaluate_position()

        self.assertEqual(evaluation["fen"], START_FEN)
        self.assertFalse(evaluation["game_over"])
        self.assertIsNone(evaluation["result"])
        self.assertIsInstance(evaluation["value"], float)
        self.assertGreaterEqual(evaluation["value"], -1.0)
        self.assertLessEqual(evaluation["value"], 1.0)

    def test_evaluate_ranks_legal_lines_by_visits(self):
        board = chess.Board()
        evaluation = self.service.evaluate_position()
        lines = evaluation["lines"]

        self.assertTrue(lines)
        visits = [line["visits"] for line in lines]
        self.assertEqual(visits, sorted(visits, reverse=True))

        for line in lines:
            self.assertIn(chess.Move.from_uci(line["move"]), board.legal_moves)
            self.assertGreater(line["visits"], 0)
            self.assertEqual(line["pv"][0], line["move"])

    def test_evaluate_takes_no_side_and_scores_black_to_move_too(self):
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        evaluation = self.service.evaluate_position(moves=["e2e4"])

        self.assertEqual(evaluation["fen"], board.fen())
        self.assertIsInstance(evaluation["value"], float)
        for line in evaluation["lines"]:
            self.assertIn(chess.Move.from_uci(line["move"]), board.legal_moves)

    def test_evaluate_pv_is_a_playable_sequence(self):
        evaluation = self.service.evaluate_position()

        for line in evaluation["lines"]:
            board = chess.Board()
            for uci in line["pv"]:
                move = chess.Move.from_uci(uci)
                self.assertIn(move, board.legal_moves)
                board.push(move)

    def test_evaluate_reports_a_won_position_on_the_result_axis(self):
        evaluation = self.service.evaluate_position(fen=CHECKMATE_FEN)

        # Same axis as `result`: -1 is a win for White.
        self.assertEqual(evaluation["value"], -1.0)
        self.assertEqual(evaluation["result"], -1)
        self.assertTrue(evaluation["game_over"])
        self.assertEqual(evaluation["lines"], [])

    def test_evaluate_reports_a_drawn_position_as_zero(self):
        evaluation = self.service.evaluate_position(fen=STALEMATE_FEN)

        self.assertEqual(evaluation["value"], 0.0)
        self.assertEqual(evaluation["result"], 0)
        self.assertTrue(evaluation["game_over"])

    def test_evaluate_rejects_an_illegal_move(self):
        with self.assertRaises(ValueError):
            self.service.evaluate_position(moves=["e2e5"])

    def test_search_finds_the_only_legal_move(self):
        # Black king on h8 is checked by the rook on h1; g8 is the only escape.
        fen = "7k/8/6K1/8/8/8/8/7R b - - 0 1"
        result = self.service.get_bot_move(fen=fen)

        self.assertEqual(result["move"], "h8g8")


if __name__ == "__main__":
    unittest.main()
