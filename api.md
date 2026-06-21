# API Reference

Base URL: `/api`

## Conventions

| Value | Meaning |
|-------|---------|
| `-1`  | X (moves first) |
| `1`   | O |
| `0`   | empty |

- Board: `3x3` integer array, `board[row][col]`.
- Action: flat index `0-8`, `index = row * 3 + col`.

## POST /api/games/{game}/move

Returns the bot's move for the given board. If the board is already terminal, no move is made.

**Path parameters**

| Name | Type | Description |
|------|------|-------------|
| `game` | string | Game id, e.g. `tictactoe`. |

**Request body**

Example:
```json
{ "board": [[-1, 1, 0], [0, -1, 0], [0, 0, 1]] }
```

**Response** `200 OK`

Example:
```json
{
  "status": 200,
  "data": { "move": 6, "game_over": false, "result": null }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `move` | int \| null | Bot's action `0-8`; `null` if the board was already terminal. |
| `game_over` | bool | `true` if the game has ended. |
| `result` | int \| null | `-1` (X), `1` (O), `0` (draw), or `null` if ongoing. |

**Errors** `400 Bad Request`

```json
{ "status": 400, "message": "unrecognized game, currently listed games are ['tictactoe']" }
```

| Cause | `message` |
|-------|-----------|
| Unknown game | `unrecognized game, ...` |
| Invalid board | marshmallow validation errors (object) |
