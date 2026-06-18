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
    board = fields.List(fields.List(fields.Integer()), required=True)

class MoveResponse(Schema):
    move = fields.Integer(allow_none=True)   # None if the board is already have a winner
    game_over = fields.Bool(required=True)
    result = fields.Integer(dump_default=None, allow_none=True)