export type GameStatus = "idle" | "human_turn" | "bot_thinking" | "win" | "lose" | "draw" | "game_over";

interface Score {
  human: number;
  bot: number;
}

interface StatusBarProps {
  status: GameStatus;
  score: Score;
  onSelectFirst: (first: "human" | "bot") => void;
}

const STATUS_TEXT: Record<GameStatus, string> = {
  idle: "",
  human_turn: "your turn",
  bot_thinking: "bot is thinking...",
  win: "you win!",
  lose: "you lose.",
  draw: "draw.",
  game_over: "game over.",
};

export function StatusBar({ status, score, onSelectFirst }: StatusBarProps) {
  return (
    <div style={{ fontFamily: "'Inter', sans-serif" }} className="flex flex-col items-center gap-4 py-4">

      <div className="flex items-center gap-8 text-3xl font-medium text-stone-700">
        <div className="flex flex-col items-center gap-1">
          <span className="text-sm font-normal text-stone-400 uppercase tracking-widest">Human</span>
          <span>{score.human === 0 ? 0 : score.human % 1 === 0 ? score.human : `${Math.floor(score.human) || ""}½`}</span>
        </div>
        <span className="text-stone-200 text-4xl font-light">|</span>
        <div className="flex flex-col items-center gap-1">
          <span className="text-sm font-normal text-stone-400 uppercase tracking-widest">Bot</span>
          <span>{score.bot === 0 ? 0 : score.bot % 1 === 0 ? score.bot : `${Math.floor(score.bot) || ""}½`}</span>
        </div>
      </div>

      {status === "idle" ? (
        <div className="flex flex-col items-center gap-2">
          <p className="text-stone-400 text-sm tracking-widest uppercase">who goes first?</p>
          <div className="flex gap-3">
            <button
              onClick={() => onSelectFirst("human")}
              className="px-6 py-2 border border-stone-300 rounded-sm text-sm tracking-widest uppercase text-stone-600 hover:bg-stone-200 hover:border-stone-400 transition-all duration-150"            >
              me
            </button>
            <button
              onClick={() => onSelectFirst("bot")}
              className="px-6 py-2 border border-stone-300 rounded-sm text-sm tracking-widest uppercase text-stone-600 hover:bg-stone-200 hover:border-stone-400 transition-all duration-150"            >
              bot
            </button>
            <button
              onClick={() => onSelectFirst(Math.random() < 0.5 ? "human" : "bot")}
              className="px-6 py-2 border border-stone-300 rounded-sm text-sm tracking-widest uppercase text-stone-600 hover:bg-stone-200 hover:border-stone-400 transition-all duration-150"            >
              random
            </button>
          </div>
        </div>
      ) : (
        <p className={`text-xl ${
          status === "win" ? "text-green-600" :
          status === "lose" ? "text-red-400" :
          status === "draw" ? "text-stone-400" :
          status === "bot_thinking" ? "text-stone-300 animate-pulse" :
          "text-stone-600"
        }`}>
          {STATUS_TEXT[status]}
        </p>
      )}

    </div>
  );
}