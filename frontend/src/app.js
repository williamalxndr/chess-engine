import { evaluatePosition } from "./api.js";
import {
  FILES,
  GLYPHS,
  findMoveByUci,
  inCheck,
  isCheckmate,
  isStalemate,
  isWhite,
  kingSquare,
  legalFrom,
  makeMove,
  san,
  startPosition,
  toUci,
} from "./chess.js";

const GAME = "chess";
const MIN_HISTORY_ROWS = 6;

const el = {
  board: document.getElementById("board"),
  status: document.getElementById("status"),
  history: document.getElementById("history"),
  engineLog: document.getElementById("engine-log"),
  evalLabel: document.getElementById("eval-label"),
  evalWhite: document.getElementById("eval-white"),
  evalBlack: document.getElementById("eval-black"),
  first: document.getElementById("first"),
  prev: document.getElementById("prev"),
  next: document.getElementById("next"),
  last: document.getElementById("last"),
  newGame: document.getElementById("new-game"),
};

let state = freshState();
// Every navigation kicks off a search; only the newest one may write to state.
let evalToken = 0;

function freshState() {
  return {
    snapshots: [startPosition()],
    moveSans: [],
    moveUcis: [],
    moveObjs: [],
    viewIndex: 0,
    selectedSq: null,
    legalTargets: [],
    legalMoveObjs: [],
    draggingSq: null,
    lines: [],
    score: null,
    evaluating: false,
    evalError: null,
    gameOver: false,
    result: null,
  };
}

const latestPos = () => state.snapshots[state.snapshots.length - 1];
const viewedPos = () => state.snapshots[state.viewIndex];
const isLatest = () => state.viewIndex === state.snapshots.length - 1;

// Analysis only: the engine never moves, so either colour can be played here.
const canPlay = () => isLatest() && !state.gameOver;

/* --- game flow ------------------------------------------------------------ */

function clearSelection() {
  state.selectedSq = null;
  state.legalTargets = [];
  state.legalMoveObjs = [];
}

function selectSquare(key) {
  const pos = latestPos();
  const piece = pos.board[key];
  if (!piece || isWhite(piece) !== (pos.turn === "w")) return false;

  const moves = legalFrom(pos, key);
  state.selectedSq = key;
  state.legalMoveObjs = moves;
  state.legalTargets = moves.map((m) => m.to);
  return true;
}

function applyMove(move) {
  const pos = latestPos();
  state.moveSans.push(san(pos, move));
  state.moveUcis.push(toUci(move));
  state.moveObjs.push(move);
  state.snapshots.push(makeMove(pos, move));
  state.viewIndex = state.snapshots.length - 1;
  clearSelection();
}

function playMove(move) {
  applyMove(move);
  state.lines = [];
  state.score = null;
  render();
  refreshEvaluation();
}

function onSquareClick(key) {
  if (!canPlay()) return;

  if (state.selectedSq) {
    // Several promotion moves share a target square; the queen is generated first.
    const move = state.legalMoveObjs.find((m) => m.to === key);
    if (move) {
      playMove(move);
      return;
    }
  }

  if (!selectSquare(key)) clearSelection();
  render();
}

/**
 * Score whatever position is on screen. Analysis takes no side, so this runs on
 * the start position, on positions being reviewed, and after either colour moves.
 */
async function refreshEvaluation() {
  const token = ++evalToken;
  const moves = state.moveUcis.slice(0, state.viewIndex);
  const pos = viewedPos();

  state.evaluating = true;
  state.evalError = null;
  render();

  try {
    const data = await evaluatePosition(GAME, { fen: null, moves });
    if (token !== evalToken) return;

    state.score = data.value;
    state.lines = data.lines.map((line) => ({
      notation: sanLine(pos, [line.move])[0] || line.move,
      score: line.value,
      variation: sanLine(pos, line.pv).join(" "),
      move: line.move,
    }));

    // Only the latest position decides whether play can continue; reviewing a
    // finished line further back must not unlock the board.
    if (isLatest()) {
      state.gameOver = data.game_over;
      state.result = data.result;
    }
  } catch (err) {
    if (token !== evalToken) return;
    state.score = null;
    state.lines = [];
    state.evalError = err.message;
  } finally {
    if (token === evalToken) {
      state.evaluating = false;
      render();
    }
  }
}

// UCI sequence to SAN, replayed through the local rules.
function sanLine(pos, ucis) {
  const out = [];
  let current = pos;

  for (const uci of ucis) {
    const move = findMoveByUci(current, uci);
    if (!move) break;
    out.push(san(current, move));
    current = makeMove(current, move);
  }

  return out;
}

function onLineClick(line) {
  if (!canPlay()) return;
  const move = findMoveByUci(latestPos(), line.move);
  if (move) playMove(move);
}

function newGame() {
  state = freshState();
  render();
  refreshEvaluation();
}

function jump(index) {
  const bounded = Math.max(0, Math.min(state.snapshots.length - 1, index));
  if (bounded === state.viewIndex) return;
  state.viewIndex = bounded;
  clearSelection();
  render();
  refreshEvaluation();
}

/* --- rendering ------------------------------------------------------------ */

// The API reports White-negative, on the same axis as `result`. Eval bars are
// read the other way round, so displayed scores are flipped.
const displayScore = (value) => -value;

const formatScore = (value) => {
  const shown = displayScore(value);
  return (shown >= 0 ? "+" : "") + shown.toFixed(2);
};

function statusText() {
  if (state.gameOver) {
    if (state.result === -1) return "Game over — White wins";
    if (state.result === 1) return "Game over — Black wins";
    return "Game over — draw";
  }

  if (!isLatest()) {
    return `Reviewing move ${state.viewIndex} of ${state.snapshots.length - 1}`;
  }

  const pos = latestPos();
  const side = pos.turn === "w" ? "White" : "Black";

  // Local mate detection only reports what the board shows; the backend is
  // still the authority on draws it cannot see (repetition, fifty-move).
  if (isCheckmate(pos)) return `Checkmate — ${pos.turn === "w" ? "Black" : "White"} wins`;
  if (isStalemate(pos)) return "Stalemate — draw";
  if (inCheck(pos, pos.turn === "w")) return `${side} to move — check`;

  return `${side} to move`;
}

function renderBoard() {
  const pos = viewedPos();
  const board = pos.board;
  const checkedKing = inCheck(pos, pos.turn === "w") ? kingSquare(board, pos.turn === "w") : null;
  const played = state.moveObjs[state.viewIndex - 1] || null;
  const fragment = document.createDocumentFragment();

  for (let rank = 8; rank >= 1; rank--) {
    for (let file = 0; file < 8; file++) {
      const key = FILES[file] + rank;
      const piece = board[key];

      const square = document.createElement("div");
      square.className = "square " + ((file + rank) % 2 === 0 ? "light" : "dark");
      if (played && (key === played.from || key === played.to)) {
        square.classList.add("last-move");
      }
      if (key === checkedKing) square.classList.add("check");
      if (key === state.selectedSq) square.classList.add("selected");

      square.addEventListener("click", () => onSquareClick(key));
      // Hover feedback toggles the class directly: re-rendering the board on
      // every dragover would tear out the node the drag is pointing at.
      square.addEventListener("dragover", (event) => {
        event.preventDefault();
        square.classList.add("drag-over");
      });
      square.addEventListener("dragleave", () => square.classList.remove("drag-over"));
      square.addEventListener("drop", (event) => {
        event.preventDefault();
        square.classList.remove("drag-over");
        state.draggingSq = null;
        onSquareClick(key);
      });

      if (piece) {
        const span = document.createElement("span");
        span.className = "piece " + (isWhite(piece) ? "white" : "black");
        if (key === state.draggingSq) span.classList.add("dragging");
        span.textContent = GLYPHS[piece];
        span.draggable = canPlay() && isWhite(piece) === (pos.turn === "w");
        span.addEventListener("dragstart", (event) => {
          if (event.dataTransfer) event.dataTransfer.effectAllowed = "move";
          // Deferred: rebuilding the board inside dragstart cancels the drag.
          setTimeout(() => {
            state.draggingSq = key;
            selectSquare(key);
            render();
          }, 0);
        });
        span.addEventListener("dragend", () => {
          state.draggingSq = null;
          render();
        });
        square.appendChild(span);
      }

      if (state.legalTargets.includes(key)) {
        const dot = document.createElement("div");
        dot.className = "dot " + (piece ? "capture" : "empty");
        square.appendChild(dot);
      }

      fragment.appendChild(square);
    }
  }

  el.board.replaceChildren(fragment);
}

function renderHistory() {
  const fragment = document.createDocumentFragment();
  const pairs = Math.max(MIN_HISTORY_ROWS, Math.ceil(state.moveSans.length / 2));

  for (let i = 0; i < pairs; i++) {
    const row = document.createElement("div");
    row.className = "history-row" + (i % 2 === 0 ? " odd" : "");

    const number = document.createElement("span");
    number.className = "number";
    number.textContent = `${i + 1}.`;

    const white = document.createElement("span");
    white.className = "white";
    white.textContent = state.moveSans[i * 2] || "";

    const black = document.createElement("span");
    black.textContent = state.moveSans[i * 2 + 1] || "";

    row.append(number, white, black);
    fragment.appendChild(row);
  }

  el.history.replaceChildren(fragment);
}

function renderLines() {
  if (!state.lines.length) {
    const empty = document.createElement("div");
    empty.className = state.evalError ? "panel-error" : "panel-empty";
    empty.textContent = state.evalError
      ? state.evalError
      : state.evaluating
        ? "Searching…"
        : "No lines to analyse in this position.";
    el.engineLog.replaceChildren(empty);
    return;
  }

  const playable = canPlay();
  const fragment = document.createDocumentFragment();

  state.lines.forEach((line) => {
    const row = document.createElement("div");
    row.className = "engine-row " + (playable ? "playable" : "inactive");
    if (playable) row.addEventListener("click", () => onLineClick(line));

    const head = document.createElement("div");
    head.className = "line";

    const notation = document.createElement("span");
    notation.className = "notation";
    notation.textContent = line.notation;

    const score = document.createElement("span");
    score.className =
      "score " + (displayScore(line.score) >= 0 ? "white-ahead" : "black-ahead");
    score.textContent = formatScore(line.score);

    head.append(notation, score);

    const variation = document.createElement("div");
    variation.className = "detail";
    variation.textContent = line.variation;

    row.append(head, variation);
    fragment.appendChild(row);
  });

  el.engineLog.replaceChildren(fragment);
}

function renderEvalBar() {
  if (state.score === null) {
    el.evalLabel.textContent = state.evaluating ? "…" : "—";
    el.evalWhite.style.height = "50%";
    el.evalBlack.style.height = "50%";
    return;
  }

  const whitePct = ((displayScore(state.score) + 1) / 2) * 100;
  el.evalLabel.textContent = formatScore(state.score);
  el.evalWhite.style.height = `${whitePct}%`;
  el.evalBlack.style.height = `${100 - whitePct}%`;
}

function render() {
  el.status.textContent = statusText();

  el.first.disabled = state.viewIndex === 0;
  el.prev.disabled = state.viewIndex === 0;
  el.next.disabled = isLatest();
  el.last.disabled = isLatest();

  renderBoard();
  renderHistory();
  renderLines();
  renderEvalBar();
}

/* --- wiring --------------------------------------------------------------- */

el.first.addEventListener("click", () => jump(0));
el.prev.addEventListener("click", () => jump(state.viewIndex - 1));
el.next.addEventListener("click", () => jump(state.viewIndex + 1));
el.last.addEventListener("click", () => jump(state.snapshots.length - 1));
el.newGame.addEventListener("click", newGame);

window.addEventListener("keydown", (event) => {
  const keys = {
    ArrowLeft: () => jump(state.viewIndex - 1),
    ArrowRight: () => jump(state.viewIndex + 1),
    ArrowUp: () => jump(0),
    ArrowDown: () => jump(state.snapshots.length - 1),
  };
  const handler = keys[event.key];
  if (!handler) return;
  event.preventDefault();
  handler();
});

render();
refreshEvaluation();
