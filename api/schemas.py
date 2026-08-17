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


class EvaluateLine(Schema):
    move = fields.String(required=True)     # UCI of the root move this line starts with
    value = fields.Float(required=True)     # same axis as `result`
    visits = fields.Integer(required=True)  # search visits behind the line
    pv = fields.List(fields.String())       # UCI moves, most-visited path from `move`

class EvaluateResponse(Schema):
    value = fields.Float(required=True)     # negative favours White, positive favours Black
    fen = fields.String(required=True)      # position that was scored
    lines = fields.List(fields.Nested(EvaluateLine))
    game_over = fields.Bool(required=True)
    result = fields.Integer(dump_default=None, allow_none=True)