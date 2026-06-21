import { Cell } from "./Cell";
import type { CellValue } from "./Cell";

interface BoardProps {
  board: CellValue[][];
  disabled: boolean;
  onCellClick: (row: number, col: number) => void;
}

export function Board({ board, disabled, onCellClick }: BoardProps) {
  return (
    <div className="inline-grid grid-cols-3 gap-0">
      {board.map((row, r) =>
        row.map((cell, c) => (
          <Cell
            key={`${r}-${c}`}
            value={cell}
            disabled={disabled || cell !== 0}
            onClick={() => onCellClick(r, c)}
          />
        ))
      )}
    </div>
  );
}