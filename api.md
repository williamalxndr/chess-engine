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

**Errors** `400 Bad Request`

```json
{ "status": 400, "message": "unrecognized game, currently listed games are ['chess']" }
```

| Cause | `message` |
|-------|-----------|
| Unknown game | `unrecognized game, ...` |
| Unparsable FEN | python-chess parse error, e.g. `expected 'w' or 'b' for turn part of fen: ...` |
| Illegal entry in `moves` | `illegal move 'e2e5' in position '...'` |
| Wrong field types | marshmallow validation errors (object) |
