from flask import Blueprint, request, jsonify
from api.schemas import SuccessResponse, ErrorResponse, MoveRequest, MoveResponse
from api.registry import GAME_REGISTRY
from api.config import LISTED_GAMES

from marshmallow import ValidationError

api_bp = Blueprint('main', __name__)

@api_bp.route("/games/<game>/move", methods=["POST"])
def move(game):
    """
    Returns {
        "move": int,
        "game_over": bool,
        "result": int,
    }
    """
    success_schema = SuccessResponse()
    error_schema = ErrorResponse()
    move_request_schema = MoveRequest()
    move_response_schema = MoveResponse()
    
    try:
        game_service = GAME_REGISTRY[game]
    except KeyError as err:
        return jsonify(
            error_schema.dump(
                {
                    "message": f"unrecognized game, currently listed games are {LISTED_GAMES}"
                }
            )
        ), 400

    try:
        payload = request.get_json()
        move_data = move_request_schema.load(payload)
    except ValidationError as err:
        return jsonify(
            error_schema.dump(
                {
                    "message": err.messages
                }
            )
        ), 400
    
    board = move_data["board"]
    bot_result = game_service.get_bot_move(board)
    move_payload = move_response_schema.dump(bot_result)

    response_body = {
        "data": move_payload
    }
    response = success_schema.dump(response_body)

    return response
