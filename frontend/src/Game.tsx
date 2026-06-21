import { useState, useEffect } from "react";

import { Board } from "@/components/Board";
import { StatusBar } from "@/components/StatusBar";
import { api } from "@/api/client";
import type { CellValue } from "@/components/Cell";
import type { GameStatus } from "@/components/StatusBar";

const EMPTY_BOARD: CellValue[][] = [
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
];

interface Score {
  human: number;
  bot: number;
}

export function Game() {
  const [board, setBoard] = useState<CellValue[][]>(EMPTY_BOARD);
  const [humanSide, setHumanSide] = useState<-1 | 1 | null>(null);
  const [gameOver, setGameOver] = useState(false);
  const [result, setResult] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [score, setScore] = useState<Score>({ human: 0, bot: 0 });

  const botSide = humanSide === 1 ? -1 : 1;

  const status: GameStatus = (() => {
    if (humanSide === null) return "idle";
    if (gameOver) {
      if (result === humanSide) return "win";
      if (result === botSide) return "lose";
      return "draw";
    }
    if (loading) return "bot_thinking";
    return "human_turn";
  })();

  useEffect(() => {
    if (humanSide === null) return;
    if (humanSide === 1) {
      runBotMove(EMPTY_BOARD);
    }
  }, [humanSide]);

  async function runBotMove(currentBoard: CellValue[][]) {
    setLoading(true);
    try {
      const res = await api.getMove({ board: currentBoard }, "tictactoe");

      if (res.move === null) {
        setGameOver(true);
        setResult(res.result);
        updateScore(res.result);
        return;
      }

      const row = Math.floor(res.move / 3);
      const col = res.move % 3;

      const newBoard = currentBoard.map((r: CellValue[], ri: number) =>
        r.map((c: CellValue, ci: number) => (ri === row && ci === col ? botSide : c))
      ) as CellValue[][];

      setBoard(newBoard);

      if (res.game_over) {
        setGameOver(true);
        setResult(res.result);
        updateScore(res.result);
      }
    } catch (err) {
      console.error("Bot move failed:", err);
    } finally {
      setLoading(false);
    }
  }

  async function handleCellClick(row: number, col: number) {
    if (gameOver || loading || board[row][col] !== 0 || humanSide === null) return;

    const boardBeforeMove = board;
    const humanBoard = board.map((r: CellValue[], ri: number) =>
      r.map((c: CellValue, ci: number) => (ri === row && ci === col ? humanSide : c))
    ) as CellValue[][];
    setBoard(humanBoard);

    setLoading(true);
    try {
      const res = await api.getMove({ board: humanBoard }, "tictactoe");

      if (res.move === null) {
        setGameOver(true);
        setResult(res.result);
        updateScore(res.result);
        return;
      }

      const botRow = Math.floor(res.move / 3);
      const botCol = res.move % 3;

      const botBoard = humanBoard.map((r: CellValue[], ri: number) =>
        r.map((c: CellValue, ci: number) => (ri === botRow && ci === botCol ? botSide : c))
      ) as CellValue[][];

      setBoard(botBoard);

      if (res.game_over) {
        setGameOver(true);
        setResult(res.result);
        updateScore(res.result);
      }
    } catch (err) {
      setBoard(boardBeforeMove);
      console.error("Move failed:", err);
    } finally {
      setLoading(false);
    }
  }

  function updateScore(res: number | null) {
    if (res === humanSide) setScore((s: Score) => ({ ...s, human: s.human + 1 }));
    else if (res === botSide) setScore((s: Score) => ({ ...s, bot: s.bot + 1 }));
    else setScore((s: Score) => ({ human: s.human + 0.5, bot: s.bot + 0.5 }));
  }

  function startNewGame(first: "human" | "bot") {
    setBoard(EMPTY_BOARD);
    setGameOver(false);
    setResult(null);
    setLoading(false);
    setHumanSide(first === "human" ? -1 : 1);
  }

  return (
    <div className="flex flex-col items-center gap-6">
      <StatusBar status={status} score={score} onSelectFirst={startNewGame} />
      <Board
        board={board}
        disabled={gameOver || loading || humanSide === null}
        onCellClick={handleCellClick}
      />
      {gameOver && (
        <button
          style={{ fontFamily: "'Inter', sans-serif" }}
          onClick={() => setHumanSide(null)}
          className="px-6 py-2 border border-stone-300 rounded-sm text-sm tracking-widest uppercase text-stone-600 hover:bg-stone-200 hover:border-stone-400 transition-all duration-150"
        >
          play again
        </button>
      )}
    </div>
  );
}