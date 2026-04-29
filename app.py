from dataclasses import dataclass
from uuid import uuid4

from flask import Flask, redirect, render_template, request, url_for

from MCTS import MCTS
from game import TicTacToe


HUMAN = -1
BOT = 1
BOT_SIMULATIONS = 1000

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
games = {}


@dataclass
class WebGame:
    env: TicTacToe
    mcts: MCTS
    current_player: int
    done: bool = False
    message: str = ""


def player_name(player):
    return {BOT: "Bot", HUMAN: "You", 0: "Draw", None: "None"}[player]


def create_game(first_player):
    env = TicTacToe()
    initial_state = env.reset()
    mcts = MCTS(
        num_simulations=BOT_SIMULATIONS,
        verbose=False,
        initial_state=initial_state,
        first_player=first_player,
        ai_player=BOT,
    )
    game = WebGame(env=env, mcts=mcts, current_player=first_player)
    game.message = "Bot is thinking..." if first_player == BOT else "You start."

    game_id = uuid4().hex
    games[game_id] = game
    return game_id


def finish_turn(game, reward, done, info):
    game.done = done
    if done:
        winner = info["winner"]
        if winner == 0:
            game.message = "It's a draw."
        elif winner == HUMAN:
            game.message = "You win."
        else:
            game.message = f"{player_name(winner)} wins."
        return

    game.current_player *= -1
    game.message = "Your turn." if game.current_player == HUMAN else "Bot is thinking..."


def bot_move(game):
    if game.done:
        return

    game.mcts.num_simulations = BOT_SIMULATIONS
    game.mcts.run()
    action = game.mcts.best_action()
    if action is None:
        legal_actions = game.mcts.legal_actions(game.env.board)
        action = legal_actions[0] if legal_actions else None
    if action is None:
        return

    state, reward, done, info = game.env.step(action, BOT)
    game.mcts.advance_to_state(state)
    finish_turn(game, reward, done, info)
    if not game.done:
        game.message = f"Bot chose {action}. Your turn."


@app.route("/")
def index():
    game_id = request.args.get("game")
    game = games.get(game_id)
    cells = []

    if game is not None:
        symbols = {0: "", BOT: "X", HUMAN: "O"}
        for action, value in enumerate(game.env.board.flatten()):
            cells.append(
                {
                    "action": action,
                    "symbol": symbols[value],
                    "disabled": game.done or game.current_player != HUMAN or value != 0,
                }
            )

    return render_template("index.html", game_id=game_id, game=game, cells=cells)


@app.route("/new", methods=["POST"])
def new_game():
    first = request.form.get("first")
    first_player = HUMAN if first == "human" else BOT
    game_id = create_game(first_player)
    return redirect(url_for("index", game=game_id))


@app.route("/move", methods=["POST"])
def move():
    game_id = request.form["game_id"]
    game = games.get(game_id)
    if game is None:
        return redirect(url_for("index"))

    if game.done or game.current_player != HUMAN:
        return redirect(url_for("index", game=game_id))

    action = int(request.form["action"])
    state, reward, done, info = game.env.step(action, HUMAN)
    if "error" in info:
        game.message = info["error"]
        return redirect(url_for("index", game=game_id))

    game.mcts.advance_to_state(state)
    finish_turn(game, reward, done, info)

    return redirect(url_for("index", game=game_id))


@app.route("/bot", methods=["POST"])
def bot_turn():
    game_id = request.form["game_id"]
    game = games.get(game_id)
    if game is None:
        return redirect(url_for("index"))

    if not game.done and game.current_player == BOT:
        bot_move(game)

    return redirect(url_for("index", game=game_id))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
