# TicTacToe MCTS

An AlphaZero-style Tic-Tac-Toe agent combining **Monte Carlo Tree Search (MCTS)** with a **policy-value neural network**, trained entirely through self-play

---

## Overview

This project replicates the core AlphaZero training loop at small scale:

1. **Self-play** generates game data using MCTS guided by a neural network
2. **Training** updates the network on collected experience via a replay buffer

**Reference:** [Mastering the Game of Go without Human Knowledge](./agz_unformatted_nature.pdf) (Silver et al., 2017)

---

## Project Structure

```
MCTS/
├── arena/
│   ├── arena.py           # Pit two agents against each other
│   ├── play.py            # Human vs. agent interface
│   └── player.py          # Player types (Human, MCTS, Random)
├── checkpoints/
│   └── *.pt               # Saved model weights
├── core/
│   ├── network.py         # Policy-value neural network 
│   └── tree.py            # MCTS tree (selection, expansion, simulation, backprop)
├── game/
│   └── env.py             # Tic-Tac-Toe environment (inheriting gym environment)
├── selfplay/
│   ├── generator.py       # Self-play game generation
│   └── replay_buffer.py   # Experience replay storage
├── training/
│   ├── pipeline.py        # Full training loop
│   └── trainer.py         # Network update logic (loss, optimizer)
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/williamalxndr/tictactoe-bot.git
cd tictactoe-bot
pip install -r requirements.txt
```

---

## How to Run

### 1. Train from scratch

Runs the full AlphaZero loop: self-play → training → arena evaluation, repeated for N iterations.

```bash
python -m training.pipeline
```

Checkpoints are saved to `checkpoints/` after each iteration.

You can configure the run with flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--path` | `version_2` | Output checkpoint name, saved in `checkpoints/<name>.pt` |
| `--iterations` | `500` | Number of train cycles |
| `--steps_per_iter` | `200` | Optimization steps per iteration |
| `--batch_size` | `64` | Batches per train step |

Example: `python -m training.pipeline --path my_run --iterations 1000`


### 2. Play against the trained agent

```bash
python -m arena.play
```

Flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--path` | `network_final` | Checkpoint to load, e.g. `iterations_200_v1.pt` |
| `--num_games` | `1` | Number of games to play |

Example: `python -m arena.play --path version_2`

Enter your move as a board position (0–8):

```
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8
```

---

## How It Works

### MCTS + Neural Network

Each move, the agent runs `N` MCTS simulations. Instead of random rollouts (random rollout approach implemented in `core.network.VanillaMCTS`), the network provides:
- **Policy head**: prior probability over legal moves (an array with size 9)
- **Value head**: absolute state value (-1 means X is 100% win, 1 means O is 100% win, and 0 means draw)

The UCB formula used for node selection:

$$U(s, a) = Q(s, a) + c_{puct} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s,b)}}{1 + N(s, a)}$$

### Training Loop

```
Self-Play  -->  Replay Buffer  -->  Network Update  -->  Arena
    ^                                                        |
    |________________________ repeat ________________________|
```

1. `generator.py` plays games using the current network + MCTS, producing `(state, policy, result)` tuples
2. `replay_buffer.py` stores and samples these tuples
3. `trainer.py` minimizes combined policy loss (cross-entropy) + value loss (MSE)

---
