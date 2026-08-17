# API Reference

Base URL: `/api`

## Conventions

| Value | Meaning |
|-------|---------|
| `-1`  | White (moves first) |
| `1`   | Black |
| `0`   | Draw |

- Position: [FEN](https://www.chessprogramming.org/Forsyth-Edwards_Notation) string.
- Move: [UCI](https://www.chessprogramming.org/Algebraic_Chess_Notation#UCI) string, e.g. `e2e4`, `e7e8q`.
- Score: float on the same axis as the table above, so `-1.0` is winning for
  White and `1.0` is winning for Black. Scores are absolute, never relative to
  the side to move — analysis takes no side.

## POST /api/games/{game}/move

Returns the bot's move for the given position. If the position is already terminal, no move is made.

The API is stateless: send the position on every request. A position is built the
same way UCI does it — start from `fen`, then replay `moves`. Sending the move
history matters for threefold-repetition draws, which a bare FEN cannot express.

**Path parameters**

| Name | Type | Description |
|------|------|-------------|
| `game` | string | Game id; only `chess` is served. |

**Request body**

| Field | Type | Description |
|-------|------|-------------|
| `fen` | string \| null | Starting position. Omit or `null` for the initial position. |
| `moves` | string[] | UCI moves replayed from `fen`. Defaults to `[]`. |

Example:
```json
{ "fen": null, "moves": ["e2e4", "e7e5"] }
```

**Response** `200 OK`

Example:
```json
{
  "status": 200,
  "data": {
    "move": "g1f3",
    "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "game_over": false,
    "result": null
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `move` | string \| null | Bot's move in UCI; `null` if the position was already terminal. |
| `fen` | string | Position after the bot's move. |
| `game_over` | bool | `true` if the game has ended. |
| `result` | int \| null | `-1` (White), `1` (Black), `0` (draw), or `null` if ongoing. |

## POST /api/games/{game}/evaluate

Scores a position without playing it, and returns the search's best root moves.
Works for any position and either side to move.

**Path parameters** and **request body** are the same as `/move`.

**Response** `200 OK`

Example:
```json
{
  "status": 200,
  "data": {
    "value": -0.31,
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "lines": [
      {
        "move": "b1c3",
        "value": -0.28,
        "visits": 45,
        "pv": ["b1c3", "a7a5", "a2a4"]
      }
    ],
    "game_over": false,
    "result": null
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `value` | float | Score for the position; negative favours White. Equals `result` when the position is terminal. |
| `fen` | string | The position that was scored. |
| `lines` | object[] | Best root moves, most-visited first. Empty when the position is terminal. |
| `game_over` | bool | `true` if the game has ended. |
| `result` | int \| null | `-1` (White), `1` (Black), `0` (draw), or `null` if ongoing. |

Each entry in `lines`:

| Field | Type | Description |
|-------|------|-------------|
| `move` | string | Root move in UCI. |
| `value` | float | Score after this move, on the same axis as `value`. |
| `visits` | int | Search visits behind the move. |
| `pv` | string[] | UCI moves along the most-visited path, starting with `move`. |

Line count and depth are capped by `EVAL_TOP_K` and `EVAL_PV_LENGTH` in `api/config.py`.

**Errors** `400 Bad Request`

Both endpoints report errors the same way:

```json
{ "status": 400, "message": "unrecognized game, currently listed games are ['chess']" }
```

| Cause | `message` |
|-------|-----------|
| Unknown game | `unrecognized game, ...` |
| Unparsable FEN | python-chess parse error, e.g. `expected 'w' or 'b' for turn part of fen: ...` |
| Illegal entry in `moves` | `illegal move 'e2e5' in position '...'` |
| Wrong field types | marshmallow validation errors (object) |
