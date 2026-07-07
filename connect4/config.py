"""Central configuration for the Connect 4 AlphaZero-style trainer.

Every value below can be overridden with an environment variable (the C4_*
names), which is how the GUI passes parameters to the trainer subprocess.
Reading them here — before any other module imports these constants — means
the overrides take effect everywhere without import-order surprises.

The stack is board-size agnostic: change the three board constants (or set
C4_ROWS/C4_COLS/C4_CONNECT) and nothing else needs to change. Defaults are
STANDARD Connect 4 (6 rows x 7 cols, four in a row).
"""

import os

import torch


def _int(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _float(name, default):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


def _str(name, default):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


# ---- Board -------------------------------------------------------------
N_ROWS = _int("C4_ROWS", 6)
N_COLS = _int("C4_COLS", 7)
N_CONNECT = _int("C4_CONNECT", 4)

# ---- Network -----------------------------------------------------------
CHANNELS = _int("C4_CHANNELS", 64)     # conv width; 64 is ample and fits easily in 16 GB
BLOCKS = _int("C4_BLOCKS", 5)          # residual blocks

# ---- MCTS --------------------------------------------------------------
SIMULATIONS = _int("C4_SIMULATIONS", 120)        # tree simulations per move (self-play)
C_PUCT = _float("C4_C_PUCT", 1.5)                # exploration constant in PUCT
DIRICHLET_ALPHA = _float("C4_DIRICHLET_ALPHA", 1.0)   # root exploration noise shape
DIRICHLET_EPS = _float("C4_DIRICHLET_EPS", 0.25)      # weight of that noise at the root
TEMP_MOVES = _int("C4_TEMP_MOVES", 10)           # first N plies sample by visits, then greedy

# ---- Self-play / training ---------------------------------------------
GAMES_PER_ITER = _int("C4_GAMES_PER_ITER", 128)      # self-play games/iteration (one batch)
WORKERS = _int("C4_WORKERS", 0)                      # parallel self-play processes (0 = auto)
REPLAY_CAPACITY = _int("C4_REPLAY_CAPACITY", 300_000)  # positions kept in the replay buffer
BATCH_SIZE = _int("C4_BATCH_SIZE", 1024)
TRAIN_STEPS_PER_ITER = _int("C4_TRAIN_STEPS", 500)
LEARNING_RATE = _float("C4_LR", 1e-3)
WEIGHT_DECAY = _float("C4_WEIGHT_DECAY", 1e-4)
ITERATIONS = _int("C4_ITERATIONS", 100_000)          # outer loop; checkpoints every iteration

# ---- Evaluation --------------------------------------------------------
# The arena is measurement only (no learning), so it runs every EVAL_EVERY
# iterations rather than every one — time better spent on self-play + training.
ARENA_GAMES = _int("C4_ARENA_GAMES", 60)         # games vs baselines to gauge strength
ARENA_SIMS = _int("C4_ARENA_SIMS", 60)           # (lighter search than self-play)
EVAL_EVERY = _int("C4_EVAL_EVERY", 10)           # run the arena every N iterations

# ---- IO / misc ---------------------------------------------------------
CHECKPOINT = _str("C4_CHECKPOINT", "connect4_az.pt")
SEED = _int("C4_SEED", 0)

_forced_device = _str("C4_DEVICE", "").lower()
if _forced_device in ("cuda", "cpu"):
    DEVICE = _forced_device
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
