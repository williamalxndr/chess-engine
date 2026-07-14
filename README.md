# Chess Zero

An AlphaZero-style chess engine combining **Monte Carlo Tree Search (MCTS)** with a **policy-value neural network**, trained through self-play

---

## Overview

This project replicates the AlphaZero training loop for chess:

1. **Self-play** generates games using MCTS guided by the current network
2. **Training** updates the network on the collected games via a replay buffer

The engine plays legal chess through [python-chess](https://python-chess.readthedocs.io). Board states are encoded as `30 × 8 × 8` planes, and moves use the AlphaZero action space (`73 × 8 × 8 = 4672` planes). The default network (`ChessNetworkV2`) is a 20-block residual network with ~24M parameters.

**Reference:** [Mastering the Game of Go without Human Knowledge](./agz_unformatted_nature.pdf) (Silver et al., 2017)

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Train through self-play

Runs the full loop: self-play → replay buffer → network update, repeated until the time or iteration budget is reached. Runs are driven by a YAML config; CLI flags override individual values.

```bash
python -m training.pipeline --config configs/local.yaml
```

Checkpoints are saved to `checkpoints/<game>/<version>/<file_name>.pt` periodically and at the end of the run.

Common flags:

| Flag | Description |
|------|-------------|
| `--version` | Network version to build/train, e.g. `V2` |
| `--file_name` | Checkpoint name to save/resume |
| `--duration` | Wall-clock training budget, in hours |
| `--num_rollout` | MCTS simulations per move |
| `--num_selfplay` | Concurrent self-play games per iteration |
| `--steps_per_iter` | Optimization steps per iteration |
| `--train_batch_size` | Training batch size |

Example: `python -m training.pipeline --version V2 --file_name example --duration 6 --num_rollout 800`

### 2. Play against the trained agent

```bash
python -m arena.play --path checkpoints/chess/V2/example.pt
```

Flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--path` | `None` | Path to a checkpoint `.pt` file |
| `--num_games` | `1` | Number of games to play |

Enter your moves in UCI notation (e.g. `e2e4`, `g1f3`).

---

## How It Works

### MCTS + Neural Network

Each move, the agent runs `num_rollout` MCTS simulations. Instead of random rollouts, the network evaluates leaf positions and returns:

- **Policy head**: prior probability over the 4672 possible moves
- **Value head**: position value in `[-1, 1]` (from the side-to-move's perspective)

Nodes are selected using the PUCT formula:

$$U(s, a) = Q(s, a) + c_{puct} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s,b)}}{1 + N(s, a)}$$

Dirichlet noise is mixed into the root priors during self-play to encourage exploration. To keep the network busy, leaf positions from many concurrent games are batched into a single forward pass.

### Self-Play Training Loop

```
Self-Play  -->  Replay Buffer  -->  Network Update
    ^                                      |
    |______________ repeat ________________|
```

1. `selfplay/generator.py` plays games with the current network + MCTS, producing `(state, policy, result)` samples, where `policy` is the MCTS visit-count distribution and `result` is the game outcome `-1 / 0 / 1`
2. `selfplay/replay_buffer.py` stores and samples these tuples
3. `training/trainer.py` minimizes combined **policy loss** (cross-entropy against the visit-count distribution, with illegal moves masked out) + **value loss** (MSE against the game result)

The improved network feeds back into the next round of self-play, and the cycle repeats.

For more details, see the [AlphaGo Zero paper](./agz_unformatted_nature.pdf).

---
