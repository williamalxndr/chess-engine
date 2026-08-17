const PREFIX = "/api";

/**
 * The API is stateless: the whole move list is replayed on every request, which
 * is what lets the backend see threefold repetition.
 */
const post = async (path, body) => {
  const response = await fetch(`${PREFIX}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    const message = payload && payload.message;
    throw new Error(
      typeof message === "string" ? message : `request failed (${response.status})`
    );
  }

  return payload.data;
};

export const evaluatePosition = (game, { fen = null, moves = [] }) =>
  post(`/games/${game}/evaluate`, { fen, moves });
