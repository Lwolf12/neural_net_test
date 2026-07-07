# Connect 4 — self-learning AI

A neural network that learns to play Connect 4 **from scratch, by playing itself**
(AlphaZero-style reinforcement learning). No perfect solver, no hand-labelled
data, no human games — the net bootstraps its own strength through self-play
guided by Monte-Carlo Tree Search (MCTS).

The project has three usable pieces:

- **Trainer** — runs the self-play → learn → repeat loop that produces the brain.
- **Training GUI** — a control panel to launch/configure the trainer and watch it improve.
- **Play GUI** — a classic Connect 4 board to play against (or watch) the trained AI.

The default board is standard Connect 4 (6 rows × 7 columns, four-in-a-row), but
the entire stack is board-size agnostic — see [Configuration](#configuration).

---

## Layout

```
neural_net_test/                 project root
├─ train_gui.py                  Tkinter control panel for training  (APP)
├─ play_gui.py                   Tkinter play-against-the-AI program  (APP)
├─ connect4_az.pt                the trained "brain" (weights + metadata; data, not code)
├─ CLAUDE.md                     this file
├─ docs/
│  └─ architecture.md            deeper dive: the RL method, MCTS, network, training math
├─ connect4/                     the engine package (importable library)
│  ├─ __init__.py                package overview
│  ├─ config.py                  all settings + board size (env-overridable)
│  ├─ game.py                    Connect 4 rules on a numpy board (no torch)
│  ├─ model.py                   the policy + value neural network
│  ├─ mcts.py                    batched Monte-Carlo Tree Search
│  ├─ selfplay.py                self-play data generation + strength evaluation
│  ├─ parallel_selfplay.py       multi-process self-play (worker pool)
│  └─ train.py                   the runnable trainer
└─ connect3_nn.py                unrelated earlier experiment (see note at the bottom)
```

**The engine lives in `connect4/` (a package); the two GUI apps live at the root**
because they import and launch that package. Run the apps from the project root
so Python can find the `connect4` package.

---

## How it works (in one picture)

```
        ┌──────────────────────────────────────────────────────┐
        │                   ONE ITERATION                       │
        │                                                       │
        │   1. SELF-PLAY        2. LEARN          3. EVAL/SAVE   │
        │   ┌───────────┐      ┌───────────┐     ┌───────────┐   │
        │   │ net + MCTS│ ───► │ gradient  │ ──► │ arena vs  │   │
        │   │ play 128  │ data │ steps on  │     │ baselines │   │
        │   │ games vs  │      │ replay    │     │ + write   │   │
        │   │ itself    │      │ buffer    │     │ checkpoint│   │
        │   └───────────┘      └───────────┘     └───────────┘   │
        │         ▲                                    │         │
        │         └────────────  repeat  ◄─────────────┘         │
        └──────────────────────────────────────────────────────┘

  one network ── policy head → which column to play
              └─ value head  → who is winning (-1 … +1)
```

Each **iteration** (~80 s on the RTX 4080): the current network plays a batch of
games against itself with MCTS lookahead; every position + the search's preferred
move + the eventual winner becomes a training example; the network is nudged to
predict those better; strength is measured and the checkpoint saved. Repeat, and
the net ratchets from random flailing to strong play.

For the *why* and the math (PUCT selection, training targets, batched search),
see **[docs/architecture.md](docs/architecture.md)**.

---

## Module reference

### `connect4/config.py`
Single source of truth for every tunable: board size, network shape, MCTS/training
hyperparameters, device, checkpoint path. Each value can be overridden by a `C4_*`
environment variable (read at import time), which is how the GUIs pass settings to
the trainer without editing code. **Imported by every other module.**

### `connect4/game.py`
The rules, with **no torch dependency** so it stays fast and testable. A board is
an `(N_ROWS, N_COLS)` int8 numpy array (`0` empty, `+1`/`-1` for the two players).
Key functions:
- `new_board()`, `legal_mask()` / `legal_actions()`, `apply_move()` (gravity drop)
- `is_win_at(board, row, col)` — checks only lines through the last move (cheap)
- `is_full()`, `encode(board, to_play)` — 2-plane network input from the mover's view
- `render()` — text board for debugging

Used by `mcts`, `selfplay`, `train`, and `play_gui`.

### `connect4/model.py`
The neural network, `Connect4Net`: a small residual conv net (AlphaZero-style).
Input is the 2-plane board; it produces **two outputs** — policy logits over the
columns (which move) and a scalar value in `[-1, 1]` (who's winning). Width/depth
come from `config` (`CHANNELS`, `BLOCKS`). Used by `mcts` (to evaluate positions),
`train` (to optimize), and `play_gui` (to pick moves).

### `connect4/mcts.py`
Batched **PUCT Monte-Carlo Tree Search** — the lookahead that makes the raw network
much stronger. `run_mcts(net, boards, to_plays, simulations, add_noise)` searches a
whole *batch* of positions at once: each game keeps its own tree, but every
simulation step collects one leaf per game and evaluates them all in a **single GPU
forward pass** (the key to using the 4080 well). Returns per-column visit counts
(the improved policy) for each position. Internals: `Node`, `_evaluate`, `_expand`,
`_select` (PUCT), `_backup`, `_add_dirichlet_noise`. Used by `selfplay` and `play_gui`.

### `connect4/selfplay.py`
Turns the net into games and games into training data.
- `play_games(net, n_games, simulations)` → self-play a batch and return
  `(samples, winners)`, where each sample is `(encoded_state, policy_target,
  value_target)`. All games advance in lockstep so their searches batch together.
- `arena(agent_a, agent_b, n_games)` → play two agents and return `(wins, draws,
  losses)`; used to measure strength.
- `net_agent`, `random_agent`, `heuristic_agent` — the players used in the arena
  (the heuristic wins-now/blocks/prefers-center is the meaningful yardstick).

Used by `train` (and `parallel_selfplay`).

### `connect4/parallel_selfplay.py`
Runs self-play across several processes so it isn't bottlenecked by
single-threaded Python MCTS. `SelfPlayPool` holds a persistent pool of spawned
worker processes, each with its own network copy; every iteration the trainer
broadcasts fresh weights, each worker plays its share of the games (reusing
`selfplay.play_games`), and returns stacked sample arrays. Workers infer on the
same GPU (idle during training); their heavy Python tree-walking overlaps across
processes, which is the actual speedup. Controlled by `C4_WORKERS`
(0 = auto ≈ CPU cores − 1, capped at 8). Used by `train`.

### `connect4/train.py`
The runnable trainer and orchestrator — the iteration loop from the picture above.
Manages the replay buffer, does the gradient steps (`_train_steps`), runs the arena,
prints progress, and checkpoints every iteration (`_save`). **Resumes automatically**
from an existing checkpoint and is **Ctrl-C safe**. It also emits machine-readable
`@@INIT@@` / `@@STATS@@` lines that the training GUI parses.
Run standalone: `python -m connect4.train`.

### `train_gui.py` (root app)
Tkinter control panel for the trainer. Lets you set all the useful parameters
(iterations, self-play games, MCTS sims, batch size, learning rate, network size,
arena settings, checkpoint, device), launches `python -m connect4.train` as a
subprocess with those as `C4_*` env vars, and shows **live stats** parsed from the
trainer's output: iteration/ETA, losses, self-play W/L/D, win-rate vs baselines, plus
two live charts (win rate and loss over iterations). Start/Stop are safe (the trainer
checkpoints every iteration).

### `play_gui.py` (root app)
Tkinter program to play Connect 4. A setup screen picks each side as **Human** or
**Computer** (all four combinations), plus AI options (checkpoint, **thinking time**,
move delay, device). AI strength is set by a **time budget** — the search runs as
many MCTS simulations as fit in the chosen thinking time, so the AI plays as strongly
as it can within that limit (`mcts.run_mcts(..., time_budget=seconds)`). **Out of the
box** it's you (Red, first) vs this project's AI at ~2 s/move; the setup screen
**remembers your last settings** (`.play_settings.json`), has a **Defaults** button to
restore them, and **Enter starts the game immediately**. The game screen is a classic
blue board with circular holes, hover preview, a falling-disc animation, click-to-drop,
and a highlighted winning line with a declared winner and **New game** / **Setup**
buttons. The AI reads its board size and network shape **from the checkpoint** (via
`EngineHolder`, which sets `C4_*` before importing the engine) so it always matches how
it was trained, and "thinks" on a background thread so the UI never freezes.

---

## Running

Python with requirements.txt installed. 

typically with the python venv (.venv) activated

```powershell
# Train from the command line
.\.venv\Scripts\python.exe -m connect4.train

# …or with the control panel
.\.venv\Scripts\python.exe train_gui.py

# Play against / watch the AI
.\.venv\Scripts\python.exe play_gui.py
```

Environment: torch is the CUDA build (`cu128`) so the RTX 4080 is used
(`torch.cuda.is_available()` is True); numpy and tkinter are installed.

---

## Configuration

All defaults live in [connect4/config.py](connect4/config.py) and each has a `C4_*`
environment-variable override (the GUIs set these for you). The most useful:

| Env var | Meaning | Default |
|---|---|---|
| `C4_ROWS` / `C4_COLS` / `C4_CONNECT` | board size + win length | 6 / 7 / 4 |
| `C4_SIMULATIONS` | MCTS sims per move in self-play | 120 |
| `C4_GAMES_PER_ITER` | self-play games per iteration | 128 |
| `C4_WORKERS` | parallel self-play processes (0 = auto) | 0 |
| `C4_EVAL_EVERY` | run the arena every N iterations | 10 |
| `C4_TRAIN_STEPS` | gradient steps per iteration | 500 |
| `C4_BATCH_SIZE` / `C4_LR` | batch size / learning rate | 1024 / 1e-3 |
| `C4_CHANNELS` / `C4_BLOCKS` | network width / depth | 64 / 5 |
| `C4_ITERATIONS` | how many iterations to run | 100000 |
| `C4_CHECKPOINT` | brain file path | `connect4_az.pt` |
| `C4_DEVICE` | `cuda` / `cpu` (blank = auto) | auto |

To switch board size, change `C4_ROWS/COLS/CONNECT` (or the constants in `config.py`)
— nothing else needs to change.

---

## The checkpoint (`connect4_az.pt`)

The trained **brain**: a PyTorch save-dict containing `model` (the learned weights),
`optimizer` (Adam state, so training resumes smoothly), `iteration` (how many
training cycles it's had), and `board` / `channels` / `blocks` (so the play GUI can
rebuild the exact network to load the weights into). The trainer **overwrites it
every iteration** — it *is* the training progress. Copy it to snapshot a good brain;
delete it (or use a new `C4_CHECKPOINT`) to train from scratch.

---

## Note on `connect3_nn.py`

`connect3_nn.py` is a separate, earlier standalone experiment: a tiny 4×3 "Connect 3"
game that is small enough to **perfectly solve** by brute force, using that perfect
solver to train a value network via supervised learning. It is unrelated to the
`connect4/` package and is kept only for reference — the Connect 4 project
deliberately uses self-play RL instead, which scales to games too large to enumerate.
