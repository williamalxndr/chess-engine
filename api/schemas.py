from marshmallow import Schema, fields

# Response wrapper DTO
class SuccessResponse(Schema):
    message = fields.String()
    status = fields.Integer(dump_default=200)
    data = fields.Raw()

class ErrorResponse(Schema):
    message = fields.Raw()
    status = fields.Integer(dump_default=400)


class MoveRequest(Schema):
    fen = fields.String(load_default=None, allow_none=True)   # None starts from the initial position
    moves = fields.List(fields.String(), load_default=list)   # UCI moves replayed from `fen`

class MoveResponse(Schema):
    move = fields.String(allow_none=True)   # UCI, None if the position was already terminal
    fen = fields.String(required=True)      # position after the bot's move
    game_over = fields.Bool(required=True)
    result = fields.Integer(dump_default=None, allow_none=True)