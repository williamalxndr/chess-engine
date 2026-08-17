from flask import Blueprint, request, jsonify
from api.schemas import (
    SuccessResponse,
    ErrorResponse,
    MoveRequest,
    MoveResponse,
    EvaluateResponse,
)
from api.registry import get_service
from api.config import LISTED_GAMES

from marshmallow import ValidationError

api_bp = Blueprint('main', __name__)

@api_bp.route("/games/<game>/move", methods=["POST"])
def move(game):
    """
    Returns {
        "move": str,
        "fen": str,
        "game_over": bool,
        "result": int,
    }
    """
    success_schema = SuccessResponse()
    error_schema = ErrorResponse()
    move_request_schema = MoveRequest()
    move_response_schema = MoveResponse()
    
    try:
        game_service = get_service(game)
    except KeyError:
        return jsonify(
            error_schema.dump(
                {
                    "message": f"unrecognized game, currently listed games are {LISTED_GAMES}"
                }
            )
        ), 400

    try:
        payload = request.get_json(silent=True) or {}
        move_data = move_request_schema.load(payload)
    except ValidationError as err:
        return jsonify(
            error_schema.dump(
                {
                    "message": err.messages
                }
            )
        ), 400

    try:
        bot_result = game_service.get_bot_move(
            fen=move_data["fen"], moves=move_data["moves"]
        )
    except ValueError as err:
        # Unparsable FEN or an illegal move in `moves`.
        return jsonify(
            error_schema.dump(
                {
                    "message": str(err)
                }
            )
        ), 400

    move_payload = move_response_schema.dump(bot_result)

    response_body = {
        "data": move_payload
    }
    response = success_schema.dump(response_body)

    return response


@api_bp.route("/games/<game>/evaluate", methods=["POST"])
def evaluate(game):
    """
    Score a position without playing it.

    Returns {
        "value": float,
        "fen": str,
        "lines": list,
        "game_over": bool,
        "result": int,
    }
    """
    success_schema = SuccessResponse()
    error_schema = ErrorResponse()
    move_request_schema = MoveRequest()
    evaluate_response_schema = EvaluateResponse()

    try:
        game_service = get_service(game)
    except KeyError:
        return jsonify(
            error_schema.dump(
                {
                    "message": f"unrecognized game, currently listed games are {LISTED_GAMES}"
                }
            )
        ), 400

    try:
        payload = request.get_json(silent=True) or {}
        move_data = move_request_schema.load(payload)
    except ValidationError as err:
        return jsonify(
            error_schema.dump(
                {
                    "message": err.messages
                }
            )
        ), 400

    try:
        evaluation = game_service.evaluate_position(
            fen=move_data["fen"], moves=move_data["moves"]
        )
    except ValueError as err:
        # Unparsable FEN or an illegal move in `moves`.
        return jsonify(
            error_schema.dump(
                {
                    "message": str(err)
                }
            )
        ), 400

    response_body = {
        "data": evaluate_response_schema.dump(evaluation)
    }

    return success_schema.dump(response_body)
