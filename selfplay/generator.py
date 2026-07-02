import numpy as np
import torch
import time
import sys
from torch import distributed as dist

from core.tree import NetworkMCTS
from core.network import PolicyValueNetwork, NetworkFactory
from core.node import Node
from core.encoder import *
from selfplay.worker import GameWorker
from core import factory
from game.rules import Rules
from profiler import timing as prof
from torch.nn.parallel import DataParallel, DistributedDataParallel
from core.factory import get_encoder

class SelfPlayGenerator:
    def __init__(self, 
                 game: str, version: str = "latest", 
                 file_name: str = None, parent_dir: str = "checkpoints", path: str = None,
                 network: PolicyValueNetwork = None,
                 num_rollout: int = 100, batch_size: int = 50, seed: int = 42):
        self.game = game
        self.version = version
        self.file_name = file_name
        self.parent_dir = parent_dir
        self.path = path
        self.num_rollout = num_rollout
        self.batch_size = batch_size
        self.rules = Rules.get(game=game)
        self.encoder = get_encoder(game, version)

        self.network = network or factory.load_network(
            path=path, game=game, version=version,
            file_name=file_name, parent_dir=parent_dir,
        )
        self.network.eval()

        self.workers: dict[int, GameWorker] = {}
        self.rng = np.random.default_rng(seed=seed)
        self._next_id = 0

    @property
    def num_worker(self):
        return len(self.workers)

    def spawn(self, n):
        for _ in range(n):
            seed = int(self.rng.integers(0, 1000000))
            self.workers[self._next_id] = GameWorker(
                game=self.game, version=self.version,
                file_name=self.file_name, parent_dir=self.parent_dir, path=self.path,
                network=self.network,
                seed=seed,
                num_rollout=self.num_rollout,
                batch_size=self.batch_size,
                add_noise=True,
            )
            self._next_id += 1

    def reset(self):
        """Drop all workers."""
        self.workers = {}
        self._next_id = 0
    
    def batch_step(self, pending: set[int]):
        """
        Run one collect + forward + backprop cycle across PENDING workers only.
        Returns dict of sims_done this batch per worker.
        """
        # ── Collect ───────────────────────────────────────────────────────────────
        t0 = time.time()

        all_eval_nodes: list[Node] = []
        offsets: dict[int, tuple[int, int, int]] = {}  # wid -> (start, end, batch_sims_done)

        for wid in pending:
            worker = self.workers[wid]
            start = len(all_eval_nodes)
            batch_nodes, batch_sims_done = worker.mcts.collect_eval_nodes()
            end = start + len(batch_nodes)
            offsets[wid] = (start, end, batch_sims_done)
            all_eval_nodes.extend(batch_nodes)

        if not all_eval_nodes:
            return {wid: data[2] for wid, data in offsets.items()}
        
        t1 = time.time()

        # ── Forward G×B ───────────────────────────────────────────────────────────
        with torch.no_grad():
            encoded_states = self.encoder.encode_batch([node.state for node in all_eval_nodes])
            rank = dist.get_rank() if dist.is_initialized() else "not a multiprocess"
            policy_head, value_head = self.network.forward(encoded_states)

        policies = policy_head.cpu().numpy()
        values = value_head.cpu().numpy()

        t2 = time.time()

        # ── Backprop ──────────────────────────────────────────────────────────────
        sims_done: dict[int, int] = {}

        for wid, (start, end, batch_sims_done) in offsets.items():
            worker_nodes = all_eval_nodes[start:end]
            worker_values = values[start:end, 0]
            worker_policies = policies[start:end]

            self.workers[wid].set_priors(worker_nodes, worker_policies)
            self.workers[wid].backpropagate(worker_nodes, worker_values)

            sims_done[wid] = len(worker_nodes) + batch_sims_done

        t3 = time.time()
        # print(f"[DEBUG] nodes: {len(all_eval_nodes)} | collect: {t1-t0:.4f}s | inference: {t2-t1:.4f}s | backprop: {t3-t2:.4f}s")


        batch_selection_time = t1 - t0
        inference_time = t2 - t1
        node_backprop_time = t3 - t2

        prof.add(batch_selection=batch_selection_time, inference=inference_time, node_backprop=node_backprop_time)

        return sims_done    
    

    def search_step(self, active_workers: set[int]):
        """
        Run a full search (num_rollout sims) for each active worker's current
        move, then sample + advance.

        Returns dict wid -> (encoded_state, policy, game_over, result).
        """
        for wid in active_workers:
            self.workers[wid].mcts._apply_root_noise()

        sims = {wid: 0 for wid in active_workers}
        pending = set(active_workers)

        while pending:
            batch_sims = self.batch_step(pending)
            for wid, n in batch_sims.items():
                sims[wid] += n
                if sims[wid] >= self.num_rollout:
                    pending.discard(wid)

        step_results = {}
        for wid in active_workers:
            worker = self.workers[wid]

            state = worker.mcts.get_current_state()
            policy = worker.get_policy()
            action = worker.sample_action(policy)

            encoded = worker.mcts.encoder.encode(state)
            legal_mask = worker.mcts.encoder.legal_action_mask(state)
            game_over, result = worker.advance(action)

            step_results[wid] = (encoded, torch.tensor(policy), game_over, result, legal_mask)
        
        print(".", end="", flush=True)

        return step_results

    def generate(self, num_games=1):
        """
        Run num_games concurrent self-play games to completion.

        Returns:
            list[tuple[list[tuple[Tensor, Tensor]], int]]:
                One entry per game. Each entry is (trajectory, result):
                - trajectory: list of (state, policy) per move
                    - state (Tensor): encoded board, shape (C, H, W)
                    - policy (Tensor): visit-count distribution, shape (action_space_size,)
                - result (int): game outcome, -1 / 0 / 1
        """
        self.network.eval()
    
        diff = num_games - self.num_worker
        if diff > 0: self.spawn(diff)

        for worker in self.workers.values():
            worker.reset()

        trajectories = {wid: [] for wid in self.workers}
        results = {}
        active_workers = set(self.workers.keys())

        while active_workers:
            step_results = self.search_step(active_workers)

            for wid, (state, policy, game_over, result, legal_mask) in step_results.items():
                trajectories[wid].append((state, policy, legal_mask))
                if game_over:
                    results[wid] = result
                    active_workers.discard(wid)

        # for wid, worker in self.workers.items():
        #     board = worker.mcts.root.state
        #     result = self.rules.get_result(board)
        #     result_str = "1-0" if result == -1 else "0-1" if result == 1 else "1/2-1/2"
        #     fen = board.fen().replace(" ", "_")
        #     moves = len(trajectories[wid])
        #     print(f"[DEBUG] Worker {wid:02d} | Moves: {moves} | Result: {result_str} | https://lichess.org/editor/{fen}")


        return [(trajectories[wid], results[wid]) for wid in trajectories]


        

if __name__ == "__main__":
    gen = SelfPlayGenerator(game="chess", num_rollout=32, batch_size=32)
    gen.generate(3)

