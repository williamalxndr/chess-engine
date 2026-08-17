export const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];

export const GLYPHS = {
  K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
  k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟",
};

const START_BOARD = {
  a8: "r", b8: "n", c8: "b", d8: "q", e8: "k", f8: "b", g8: "n", h8: "r",
  a7: "p", b7: "p", c7: "p", d7: "p", e7: "p", f7: "p", g7: "p", h7: "p",
  a2: "P", b2: "P", c2: "P", d2: "P", e2: "P", f2: "P", g2: "P", h2: "P",
  a1: "R", b1: "N", c1: "B", d1: "Q", e1: "K", f1: "B", g1: "N", h1: "R",
};

export const startPosition = () => ({
  board: { ...START_BOARD },
  turn: "w",
  castling: "KQkq",
  ep: null,
});

export const isWhite = (piece) => !!piece && piece === piece.toUpperCase();

const inBoard = (f, r) => f >= 0 && f <= 7 && r >= 1 && r <= 8;
const square = (f, r) => FILES[f] + r;
const fileIndex = (sq) => FILES.indexOf(sq[0]);
const rankIndex = (sq) => parseInt(sq[1], 10);

export const kingSquare = (board, white) => {
  const target = white ? "K" : "k";
  return Object.keys(board).find((key) => board[key] === target) || null;
};

// True if `target` is attacked by the given side. Used both for check detection
// and for the squares a king may not cross while castling.
export const attacked = (board, target, byWhite) => {
  const tf = fileIndex(target);
  const tr = rankIndex(target);

  const pawnDir = byWhite ? -1 : 1;
  for (const df of [-1, 1]) {
    const f = tf + df;
    const r = tr + pawnDir;
    if (inBoard(f, r) && board[square(f, r)] === (byWhite ? "P" : "p")) return true;
  }

  const knightHops = [[1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2]];
  for (const [df, dr] of knightHops) {
    const f = tf + df;
    const r = tr + dr;
    if (inBoard(f, r) && board[square(f, r)] === (byWhite ? "N" : "n")) return true;
  }

  const kingSteps = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
  for (const [df, dr] of kingSteps) {
    const f = tf + df;
    const r = tr + dr;
    if (inBoard(f, r) && board[square(f, r)] === (byWhite ? "K" : "k")) return true;
  }

  const rays = [
    { dirs: [[1, 0], [-1, 0], [0, 1], [0, -1]], pieces: byWhite ? ["R", "Q"] : ["r", "q"] },
    { dirs: [[1, 1], [1, -1], [-1, 1], [-1, -1]], pieces: byWhite ? ["B", "Q"] : ["b", "q"] },
  ];
  for (const ray of rays) {
    for (const [df, dr] of ray.dirs) {
      let f = tf + df;
      let r = tr + dr;
      while (inBoard(f, r)) {
        const occupant = board[square(f, r)];
        if (occupant) {
          if (ray.pieces.includes(occupant)) return true;
          break;
        }
        f += df;
        r += dr;
      }
    }
  }

  return false;
};

export const inCheck = (pos, white) => {
  const king = kingSquare(pos.board, white);
  return king ? attacked(pos.board, king, !white) : false;
};

// Moves ignoring whether the mover's own king is left in check.
const pseudoFrom = (pos, from) => {
  const board = pos.board;
  const piece = board[from];
  if (!piece) return [];

  const white = isWhite(piece);
  const type = piece.toUpperCase();
  const f0 = fileIndex(from);
  const r0 = rankIndex(from);
  const out = [];

  const push = (to, extra) => out.push({ from, to, piece, ...(extra || {}) });

  const step = (f, r) => {
    if (!inBoard(f, r)) return false;
    const to = square(f, r);
    const occupant = board[to];
    if (occupant) {
      if (isWhite(occupant) !== white) push(to, { capture: occupant });
      return false;
    }
    push(to);
    return true;
  };

  // Queen first: the UI takes the first move matching a target square, and a
  // promotion picker is not part of the design.
  const promoPieces = white ? ["Q", "R", "B", "N"] : ["q", "r", "b", "n"];

  if (type === "P") {
    const dir = white ? 1 : -1;
    const startRank = white ? 2 : 7;
    const lastRank = white ? 8 : 1;

    const one = square(f0, r0 + dir);
    if (inBoard(f0, r0 + dir) && !board[one]) {
      if (r0 + dir === lastRank) promoPieces.forEach((promo) => push(one, { promo }));
      else push(one);

      const two = square(f0, r0 + 2 * dir);
      if (r0 === startRank && !board[two]) push(two, { double: true });
    }

    for (const df of [-1, 1]) {
      const f = f0 + df;
      const r = r0 + dir;
      if (!inBoard(f, r)) continue;
      const to = square(f, r);
      const occupant = board[to];
      if (occupant && isWhite(occupant) !== white) {
        if (r === lastRank) {
          promoPieces.forEach((promo) => push(to, { capture: occupant, promo }));
        } else {
          push(to, { capture: occupant });
        }
      } else if (!occupant && pos.ep === to) {
        push(to, { ep: true, capture: white ? "p" : "P" });
      }
    }
  } else if (type === "N") {
    [[1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2]]
      .forEach(([df, dr]) => step(f0 + df, r0 + dr));
  } else if (type === "K") {
    [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
      .forEach(([df, dr]) => step(f0 + df, r0 + dr));

    const rank = white ? 1 : 8;
    const rights = pos.castling || "";
    const kingHome = "e" + rank;
    if (from === kingHome && !attacked(board, kingHome, !white)) {
      const shortRight = white ? "K" : "k";
      const longRight = white ? "Q" : "q";
      if (
        rights.includes(shortRight) &&
        !board["f" + rank] && !board["g" + rank] &&
        !attacked(board, "f" + rank, !white) && !attacked(board, "g" + rank, !white)
      ) {
        push("g" + rank, { castle: "K" });
      }
      if (
        rights.includes(longRight) &&
        !board["d" + rank] && !board["c" + rank] && !board["b" + rank] &&
        !attacked(board, "d" + rank, !white) && !attacked(board, "c" + rank, !white)
      ) {
        push("c" + rank, { castle: "Q" });
      }
    }
  } else {
    const dirs =
      type === "B" ? [[1, 1], [1, -1], [-1, 1], [-1, -1]]
      : type === "R" ? [[1, 0], [-1, 0], [0, 1], [0, -1]]
      : [[1, 1], [1, -1], [-1, 1], [-1, -1], [1, 0], [-1, 0], [0, 1], [0, -1]];

    dirs.forEach(([df, dr]) => {
      let f = f0 + df;
      let r = r0 + dr;
      while (step(f, r)) {
        f += df;
        r += dr;
      }
    });
  }

  return out;
};

export const makeMove = (pos, move) => {
  const board = { ...pos.board };
  const white = isWhite(move.piece);
  const rank = white ? 1 : 8;

  delete board[move.from];
  board[move.to] = move.promo ? move.promo : move.piece;

  if (move.ep) {
    delete board[move.to[0] + (white ? rankIndex(move.to) - 1 : rankIndex(move.to) + 1)];
  }
  if (move.castle === "K") {
    delete board["h" + rank];
    board["f" + rank] = white ? "R" : "r";
  }
  if (move.castle === "Q") {
    delete board["a" + rank];
    board["d" + rank] = white ? "R" : "r";
  }

  let castling = pos.castling || "";
  if (move.piece.toUpperCase() === "K") {
    castling = castling.replace(white ? /[KQ]/g : /[kq]/g, "");
  }
  if (move.from === "h1" || move.to === "h1") castling = castling.replace("K", "");
  if (move.from === "a1" || move.to === "a1") castling = castling.replace("Q", "");
  if (move.from === "h8" || move.to === "h8") castling = castling.replace("k", "");
  if (move.from === "a8" || move.to === "a8") castling = castling.replace("q", "");

  const ep = move.double
    ? move.to[0] + (rankIndex(move.from) + rankIndex(move.to)) / 2
    : null;

  return { board, turn: white ? "b" : "w", castling, ep };
};

export const legalFrom = (pos, from) => {
  const white = isWhite(pos.board[from]);
  return pseudoFrom(pos, from).filter((m) => !inCheck(makeMove(pos, m), white));
};

export const allLegal = (pos) => {
  const white = pos.turn === "w";
  const out = [];
  Object.keys(pos.board).forEach((from) => {
    if (isWhite(pos.board[from]) === white) out.push(...legalFrom(pos, from));
  });
  return out;
};

export const san = (pos, move) => {
  const next = makeMove(pos, move);
  const check = inCheck(next, next.turn === "w");
  const noReplies = allLegal(next).length === 0;
  const suffix = check ? (noReplies ? "#" : "+") : "";

  if (move.castle) return (move.castle === "K" ? "O-O" : "O-O-O") + suffix;

  const promo = move.promo ? "=" + move.promo.toUpperCase() : "";
  const type = move.piece.toUpperCase();

  if (type === "P") {
    const body = move.capture ? move.from[0] + "x" + move.to : move.to;
    return body + promo + suffix;
  }

  // Disambiguate only against pieces of the same type that could reach the same square.
  const rivals = allLegal(pos).filter(
    (o) => o.piece === move.piece && o.to === move.to && o.from !== move.from
  );
  let disambiguation = "";
  if (rivals.length) {
    const sameFile = rivals.some((o) => o.from[0] === move.from[0]);
    const sameRank = rivals.some((o) => o.from[1] === move.from[1]);
    disambiguation = sameFile && sameRank ? move.from : sameFile ? move.from[1] : move.from[0];
  }

  return type + disambiguation + (move.capture ? "x" : "") + move.to + suffix;
};

export const toUci = (move) => move.from + move.to + (move.promo ? move.promo.toLowerCase() : "");

export const findMoveByUci = (pos, uci) => allLegal(pos).find((m) => toUci(m) === uci) || null;

export const isCheckmate = (pos) => inCheck(pos, pos.turn === "w") && allLegal(pos).length === 0;

export const isStalemate = (pos) => !inCheck(pos, pos.turn === "w") && allLegal(pos).length === 0;
