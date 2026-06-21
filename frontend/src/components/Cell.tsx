export type CellValue = -1 | 1 | 0;

const CELL_DISPLAY: Record<CellValue, string> = {
  [-1]: "X",
  1: "O",
  0: "",
};

interface CellProps {
  value: CellValue;
  disabled: boolean;
  onClick?: () => void;
}

export function Cell({ value, disabled, onClick }: CellProps) {
  return (
    <div
      onClick={!disabled ? onClick : undefined}
      style={{ fontFamily: "'Patrick Hand', cursive" }}
      className={`
        w-20 h-20 flex items-center justify-center
        border border-stone-300 rounded-sm
        text-5xl
        bg-[#faf8f3] text-stone-800
        transition-colors duration-100
        ${disabled ? "opacity-60 cursor-not-allowed" : "cursor-pointer hover:bg-stone-200"}
      `}
    >
      {CELL_DISPLAY[value]}
    </div>
  );
}