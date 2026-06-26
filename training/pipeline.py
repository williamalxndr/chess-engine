import time
import numpy as np
import torch
from torch import optim
import copy
import argparse
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from pathlib import Path
from contextlib import contextmanager

from core.network import PolicyValueNetwork
from selfplay.replay_buffer import ReplayBuffer
from selfplay.generator import SelfPlayGenerator
from training.trainer import Trainer
from core import config


class Pipeline:
    """
    Generator  -->  ReplayBuffer  -->  Trainer
   (self-play)     (stores s,pi,z)   (optimizes net)
        ^                                  |
        |_______ improved network _________|
    """

    def __init__(self, game: str = "chess", version: str = "latest",
                 load_file_name: str = None, save_file_name: str = None,
                 parent_dir: str = "checkpoints", path: str = None,
                 buffer_file_name: str = None, buffer_path: str = None,
                 optimizer: optim.Adam = None,
                 train_batch_size: int = 32,
                 replay_buffer_max_size: int = 10000,
                 iterations: int = 50,
                 mcts_batch_size: int = 16,
                 num_rollout: int = 100,
                 num_selfplay: int = 2,
                 steps_per_iter: int = 200,
                 patience: int = 50,
                 verbose: bool = False,
                 kaggle: bool = False):

        self.game             = game
        self.version          = version
        self.save_file_name   = save_file_name or load_file_name or "fallback"
        self.parent_dir       = parent_dir
        self.train_batch_size = train_batch_size
        self.iterations       = iterations
        self.steps_per_iter   = steps_per_iter
        self.num_selfplay     = num_selfplay
        self.patience         = patience
        self.verbose          = verbose
        self.kaggle           = kaggle

        # Network
        self.network = config.load_network(
            path=path, game=game, version=version,
            file_name=load_file_name, parent_dir=parent_dir,
        )

        # Replay buffer
        _buffer_path = Path(f"{parent_dir}/{game}/{version}/{buffer_file_name}_buffer.pt") if buffer_file_name else None

        if buffer_path is not None:
            self.replay_buffer = ReplayBuffer.load(path=buffer_path)
            print("Buffer loaded from path")
        elif _buffer_path and _buffer_path.is_file():
            self.replay_buffer = ReplayBuffer.load(game=game, version=version, file_name=buffer_file_name, parent_dir=parent_dir)
            print(f"Buffer loaded from {parent_dir}")
        else:
            self.replay_buffer = ReplayBuffer(replay_buffer_max_size)
            print("Buffer created")

        self.generator = SelfPlayGenerator(
            game=game, version=version,
            file_name=load_file_name, parent_dir=parent_dir, path=path,
            num_rollout=num_rollout, batch_size=mcts_batch_size,
        )
        self.trainer = Trainer(
            self.network,
            optim.Adam(self.network.parameters(), lr=0.01, fused=True) if optimizer is None else optimizer,
            T_max=iterations,
        )


    # ── Core loop ─────────────────────────────────────────────────────────────

    def generate(self):
        results = self.generator.generate(self.num_selfplay)
        for trajectory, z in results:
            self.replay_buffer.add(trajectory, z)

    def sample(self):
        return self.replay_buffer.sample(self.train_batch_size)

    def train_step(self):
        s, pi, z = self.sample()
        return self.trainer.step(s, pi, z)

    def train(self, duration_hours=None, log_interval=1):
        start = time.time()

        duration_seconds = duration_hours * 3600 if duration_hours else None

        with self._make_progress(duration_hours) as (progress, task):
            iteration      = 0
            start_time     = time.time()
            last_save_time = start_time
            save_interval  = 600
            min_loss       = float('inf')
            not_improving  = 0
            loss = policy_loss = v_loss = 0.0

            while True:
                current_time = time.time()

                if self._should_stop(current_time, start_time, duration_seconds, iteration):
                    break

                self.network.eval()
                self.generate()
                while len(self.replay_buffer) < self.train_batch_size:
                    self.generate()

                self.network.train()
                for _ in range(self.steps_per_iter):
                    loss, policy_loss, v_loss = self.train_step()

                min_loss, not_improving = self._update_early_stopping(loss, min_loss, not_improving)
                if not_improving >= self.patience:
                    break

                current_time = time.time()

                if self.kaggle and iteration % log_interval == 0:
                    elapsed = current_time - start_time
                    remaining_str = ""
                    if duration_seconds:
                        remaining = max(0, duration_seconds - elapsed)
                        remaining_str = f" | {remaining/3600:.2f}h left"
                    patience_str = f" | patience: {not_improving}/{self.patience}" if not_improving > self.patience * 0.5 else ""
                    print(f"iter {iteration} | loss: {loss:.4f} | policy: {policy_loss:.4f} | value: {v_loss:.4f} | {elapsed/60:.1f}m elapsed{remaining_str}{patience_str} | replay buffer: {len(self.replay_buffer)}")

                # Save every 10 minutes
                if current_time - last_save_time >= save_interval:
                    self.save()
                    last_save_time = current_time

                self._update_progress(progress, task, current_time, start_time, duration_seconds, loss, policy_loss, v_loss, not_improving)
                self.trainer.scheduler.step()
                iteration += 1

        end = time.time()
        elapsed = time.time() - start_time
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)

        self.save()

        parent_dir = "kaggle" if self.kaggle else "checkpoints"
        print(f"\nTraining finished after {h}h {m}m {s}s! To play against the trained network, run:")
        print(f"  python3 -m arena.play --game {self.game} --version {self.version} --file_name {self.save_file_name}")
        print(f"  OR")
        print(f"  python3 -m arena.play --path {parent_dir}/{self.game}/{self.version}/{self.save_file_name}.pt")       
        
        return self.get_network()


    # ── Helpers ───────────────────────────────────────────────────────────────

    @contextmanager
    def _make_progress(self, duration_hours):
        initial_desc = "loss: 0.0000 | policy loss: 0.0000 | value loss: 0.0000"
        if duration_hours:
            initial_desc = f"Time left: {duration_hours:.2f}h | " + initial_desc

        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            transient=False,
            disable=not self.verbose or self.kaggle,
        )
        task = progress.add_task(initial_desc, total=self.iterations)
        with progress:
            yield progress, task

    def _should_stop(self, current_time, start_time, duration_seconds, iteration):
        if duration_seconds and (current_time - start_time) >= duration_seconds:
            return True
        return iteration >= self.iterations

    def _update_early_stopping(self, loss, min_loss, not_improving):
        if loss < min_loss:
            return loss, 0
        return min_loss, not_improving + 1

    def _update_progress(self, progress, task, current_time, start_time,
                         duration_seconds, loss, policy_loss, v_loss, not_improving):
        if not self.verbose or self.kaggle:
            return
        patience_str = f" | ⚠ patience: {not_improving}/{self.patience}" if not_improving > self.patience * 0.5 else ""
        desc = f"loss: {loss:.4f} | policy loss: {policy_loss:.4f} | value loss: {v_loss:.4f}{patience_str}"
        if duration_seconds:
            remaining = max(0, duration_seconds - (current_time - start_time))
            desc = f"Time left: {remaining/3600:.2f}h | " + desc
        progress.update(task, advance=1, description=desc)

    def get_network(self):
        return copy.deepcopy(self.network)

    def save(self, file_name: str = None):
        file_name  = file_name or self.save_file_name
        parent_dir = "kaggle" if self.kaggle else "checkpoints"
        self.network.save(self.game, self.version, file_name=file_name, parent_dir=parent_dir)
        self.replay_buffer.save(self.game, self.version, file_name=file_name, parent_dir=parent_dir)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game",                   type=str,   default="chess")
    parser.add_argument("--version",                type=str,   default="V2")
    parser.add_argument("--file_name",              type=str,   default=None)
    parser.add_argument("--iterations",             type=int,   default=99999)
    parser.add_argument("--steps_per_iter",         type=int,   default=200)
    parser.add_argument("--train_batch_size",       type=int,   default=32)
    parser.add_argument("--mcts_batch_size",        type=int,   default=16)
    parser.add_argument("--num_rollout",            type=int,   default=100)
    parser.add_argument("--num_selfplay",           type=int,   default=2)
    parser.add_argument("--duration",               type=float, default=None)
    parser.add_argument("--verbose",                action="store_true", default=False)
    parser.add_argument("--kaggle",                 action="store_true", default=False)
    parser.add_argument("--log_interval",           type=int,   default=1)
    parser.add_argument("--replay_buffer_max_size", type=int,   default=10000)
    parser.add_argument("--kaggle_network",         type=str,   default=None)
    parser.add_argument("--kaggle_buffer",          type=str,   default=None)
    args = parser.parse_args()

    # ── Resolve network path ──────────────────────────────────────────────────
    kaggle_network_path = Path(f"/kaggle/input/{args.kaggle_network}/{args.file_name}.pt") if args.kaggle_network else None
    local_network_path  = Path(f"checkpoints/{args.game}/{args.version}/{args.file_name}.pt") if args.file_name else None

    if kaggle_network_path and kaggle_network_path.is_file():
        network_path, network_file_name = str(kaggle_network_path), None
        print(f"Network: Kaggle {kaggle_network_path}")
    elif local_network_path and local_network_path.is_file():
        network_path, network_file_name = None, args.file_name
        print(f"Network loaded from local checkpoint {local_network_path}")
    else:
        network_path, network_file_name = None, None
        print("Network created")

    # ── Resolve buffer path ───────────────────────────────────────────────────
    kaggle_buffer_path = Path(f"/kaggle/input/{args.kaggle_buffer}/{args.file_name}_buffer.pt") if args.kaggle_buffer else None

    if kaggle_buffer_path and kaggle_buffer_path.is_file():
        buffer_path, buffer_file_name = str(kaggle_buffer_path), None
        print(f"Buffer: Kaggle {kaggle_buffer_path}")
    else:
        buffer_path, buffer_file_name = None, args.file_name

    # ── Run ───────────────────────────────────────────────────────────────────
    pipeline = Pipeline(
        game=args.game,
        version=args.version,
        load_file_name=network_file_name,
        save_file_name=args.file_name,
        path=network_path,
        buffer_file_name=buffer_file_name,
        buffer_path=buffer_path,
        iterations=args.iterations,
        steps_per_iter=args.steps_per_iter,
        train_batch_size=args.train_batch_size,
        mcts_batch_size=args.mcts_batch_size,
        num_rollout=args.num_rollout,
        num_selfplay=args.num_selfplay,
        replay_buffer_max_size=args.replay_buffer_max_size,
        verbose=args.verbose,
        kaggle=args.kaggle,
    )

    pipeline.train(
        duration_hours=args.duration,
        log_interval=args.log_interval,
    )