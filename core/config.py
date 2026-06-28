import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class MCTSConfig:
    num_rollout: int = 64
    mcts_batch_size: int = 8
    num_selfplay: int = 4


@dataclass
class TrainingConfig:
    duration: Optional[float] = None
    steps_per_iter: int = 64
    train_batch_size: int = 16
    replay_buffer_max_size: int = 10000


@dataclass
class LoggingConfig:
    kaggle: bool = False
    verbose: bool = False
    log_interval: int = 1


@dataclass
class KaggleDatasets:
    network: Optional[str] = None
    buffer: Optional[str] = None


@dataclass
class Config:
    game: str = "chess"
    version: str = "V2"
    file_name: str = "test"

    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    kaggle_datasets: KaggleDatasets = field(default_factory=KaggleDatasets)

    def summary(self) -> str:
        lines = [
            f"  game={self.game}  version={self.version}  file={self.file_name}",
            f"  mcts    : rollouts={self.mcts.num_rollout}  batch={self.mcts.mcts_batch_size}  selfplay={self.mcts.num_selfplay}",
            f"  training: duration={self.training.duration}h  steps={self.training.steps_per_iter}  batch={self.training.train_batch_size}  buffer={self.training.replay_buffer_max_size}",
            f"  logging : kaggle={self.logging.kaggle}  verbose={self.logging.verbose}  log_interval={self.logging.log_interval}",
        ]
        if self.kaggle_datasets.network:
            lines.append(f"  datasets: network={self.kaggle_datasets.network}")
        if self.kaggle_datasets.buffer:
            lines.append(f"            buffer={self.kaggle_datasets.buffer}")
        return "\n".join(lines)


def _find_repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(6):
        if (p / "configs").is_dir() or (p / ".git").exists():
            return p
        p = p.parent
    return Path.cwd()


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _apply_cli_overrides(cfg: Config, overrides: argparse.Namespace) -> Config:
    mapping = {
        "game":                     ("root",     "game"),
        "version":                  ("root",     "version"),
        "file_name":                ("root",     "file_name"),
        "num_rollout":              ("mcts",     "num_rollout"),
        "mcts_batch_size":          ("mcts",     "mcts_batch_size"),
        "num_selfplay":             ("mcts",     "num_selfplay"),
        "duration":                 ("training", "duration"),
        "steps_per_iter":           ("training", "steps_per_iter"),
        "train_batch_size":         ("training", "train_batch_size"),
        "replay_buffer_max_size":   ("training", "replay_buffer_max_size"),
        "kaggle":                   ("logging",  "kaggle"),
        "verbose":                  ("logging",  "verbose"),
        "log_interval":             ("logging",  "log_interval"),
        "kaggle_network":           ("datasets", "network"),
        "kaggle_buffer":            ("datasets", "buffer"),
    }
    for cli_key, (section, attr) in mapping.items():
        val = getattr(overrides, cli_key, None)
        if val is None:
            continue
        if section == "root":
            setattr(cfg, attr, val)
        elif section == "mcts":
            setattr(cfg.mcts, attr, val)
        elif section == "training":
            setattr(cfg.training, attr, val)
        elif section == "logging":
            setattr(cfg.logging, attr, val)
        elif section == "datasets":
            setattr(cfg.kaggle_datasets, attr, val)
    return cfg


def load_config(argv: list[str] = None) -> Config:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config",                 type=str,   default=None)
    parser.add_argument("--game",                   type=str,   default=None)
    parser.add_argument("--version",                type=str,   default=None)
    parser.add_argument("--file_name",              type=str,   default=None)
    parser.add_argument("--duration",               type=float, default=None)
    parser.add_argument("--num_rollout",            type=int,   default=None)
    parser.add_argument("--mcts_batch_size",        type=int,   default=None)
    parser.add_argument("--num_selfplay",           type=int,   default=None)
    parser.add_argument("--steps_per_iter",         type=int,   default=None)
    parser.add_argument("--train_batch_size",       type=int,   default=None)
    parser.add_argument("--replay_buffer_max_size", type=int,   default=None)
    parser.add_argument("--log_interval",           type=int,   default=None)
    parser.add_argument("--kaggle",                 action="store_true", default=None)
    parser.add_argument("--verbose",                action="store_true", default=None)
    parser.add_argument("--kaggle_network",         type=str,   default=None)
    parser.add_argument("--kaggle_buffer",          type=str,   default=None)

    args, _ = parser.parse_known_args(argv)

    root = _find_repo_root()
    if args.config:
        yaml_path = Path(args.config)
        if not yaml_path.is_absolute():
            yaml_path = root / yaml_path
    else:
        yaml_path = root / "configs" / "local.yaml"

    if not yaml_path.exists():
        available = [p.name for p in (root / "configs").glob("*.yaml")]
        raise FileNotFoundError(f"Config not found: {yaml_path}\nAvailable: {available}")

    raw = _load_yaml(yaml_path)

    cfg = Config(
        game      = raw.get("game", "chess"),
        version   = raw.get("version", "V2"),
        file_name = raw.get("file_name", "test"),
        mcts      = MCTSConfig(**raw.get("mcts", {})),
        training  = TrainingConfig(**raw.get("training", {})),
        logging   = LoggingConfig(**raw.get("logging", {})),
        kaggle_datasets = KaggleDatasets(**raw.get("kaggle_datasets", {})),
    )

    return _apply_cli_overrides(cfg, args)