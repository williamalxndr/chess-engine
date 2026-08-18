LISTED_GAMES = ["chess"]

# Checkpoint served by the API. CHECKPOINT_PATH names a file directly and wins
# when set; otherwise the file is addressed as
# <PARENT_DIR>/<game>/<VERSION>/<FILE_NAME>.pt
CHECKPOINT_PATH = "pretrained-network/pretrained.pt"
CHECKPOINT_VERSION = "V2"
CHECKPOINT_FILE_NAME = "test"
CHECKPOINT_PARENT_DIR = "checkpoints"

SEED = 42
NUM_ROLLOUT = 100
MCTS_BATCH_SIZE = 8

# Analysis: how many root moves to report, and how deep to follow each one.
EVAL_TOP_K = 4
EVAL_PV_LENGTH = 8
