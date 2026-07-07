# Architecture & algorithm deep-dive

Companion to the root [CLAUDE.md](../CLAUDE.md). This explains *why* the Connect 4
AI is built the way it is and how the learning actually works. It assumes you've
read the overview there.

---

## Why self-play RL instead of a solver

Connect 4 has far too many positions to enumerate and label perfectly on any real
board, so we can't just compute the right answer for every position (that only works
for tiny games — see `connect3_nn.py`). Instead the network learns the way AlphaZero
does: it **plays itself**, and the outcomes of those games are the only supervision.
There is no external teacher — strength is bootstrapped. Weak play still produces a
useful training signal (whoever happened to win), the network improves slightly, and
better networks generate better games to learn from. That feedback loop is the whole
method.

---

## The network (`model.py`)

`Connect4Net` is a small residual convolutional network with **two heads**:

```
board (2 planes: my pieces, their pieces)
        │
     stem conv → N residual blocks        (spatial feature extractor)
        │
        ├── policy head → logits over columns   "which move looks good?"
        └── value head  → tanh scalar in [-1,1]  "who is winning, from side-to-move?"
```

- **Input encoding** (`game.encode`) is always from the perspective of the player to
  move: plane 0 = my pieces, plane 1 = opponent pieces. The net therefore only learns
  one "me vs them" view instead of a separate view per color, which halves what it has
  to learn and makes both sides share knowledge.
- **Two planes, one board size.** Width (`CHANNELS`) and depth (`BLOCKS`) come from
  `config`; 64×5 is ample for Connect 4 and tiny for a 16 GB GPU.

---

## The search: batched PUCT MCTS (`mcts.py`)

The raw network is a fast but shallow judge. MCTS turns it into a much stronger player
by doing **guided lookahead**: it grows a search tree, using the network to decide
which lines are worth exploring and to estimate leaf positions, then plays the move it
ended up visiting most.

### One simulation
Each simulation walks from the root down to a leaf, choosing at every node the child
that maximizes the **PUCT** score:

```
score(child) =  Q(child)              +  C_PUCT · P(child) · √(N_parent) / (1 + N_child)
                └ exploitation ┘         └────────────── exploration ──────────────┘
```

- `Q` is the child's average value seen so far (from that node's perspective; the
  parent negates it, since players alternate).
- `P` is the network's prior probability for that move.
- `N` are visit counts. Unvisited, network-favored moves get explored first; as a
  move accumulates visits its exploration term shrinks.

At the leaf, the network evaluates the position (prior + value). The value is then
**backed up** the path, flipping sign at each level (a good position for me is a bad
one for my opponent). Terminal positions (a completed line or a full board) use their
exact result instead of the network.

After `SIMULATIONS` simulations, the **visit-count distribution** over the root's
moves is the search's improved policy — stronger than the network's raw policy, and
what both self-play (as a training target) and the play GUI (to pick a move) use.

### Why batched, and why it matters for the GPU
The naive implementation evaluates one leaf at a time — thousands of tiny GPU calls,
each dominated by launch overhead. Instead, `run_mcts` searches **many games at once**:
every game has its own tree, but on each simulation step it gathers one leaf per game
and runs a **single batched forward pass** over all of them. That converts thousands
of tiny evaluations into a handful of large ones and is what actually keeps the RTX
4080 busy. The practical cost that remains is the Python tree-walking on the CPU, which
is why per-iteration time scales with `SIMULATIONS × GAMES_PER_ITER`, not GPU horsepower.

### Root exploration noise
During self-play (not during evaluation or real games), Dirichlet noise is mixed into
the root priors (`DIRICHLET_ALPHA`, `DIRICHLET_EPS`). This forces the AI to occasionally
try moves it currently underrates, so training data keeps covering the whole game rather
than collapsing onto one opening.

---

## Self-play → training data (`selfplay.py`)

`play_games` plays a batch of games to completion. All games advance one ply at a time
in lockstep, so at any moment every still-running game shares the same side to move —
which is exactly what lets one `run_mcts` call serve the whole batch.

For every move in every game it records:

| part of a training sample | value |
|---|---|
| **state** | `encode(board, side_to_move)` — the 2-plane input |
| **policy target** | the MCTS visit-count distribution over columns |
| **value target** | filled in at game end: `+1` if that side went on to win, `-1` if it lost, `0` for a draw |

Early moves (first `TEMP_MOVES` plies) are chosen by *sampling* from the visit counts
(temperature = 1) for opening variety; later moves are played greedily (the most-visited
move).

---

## Training (`train.py`)

Samples flow into a fixed-size **replay buffer** (recent positions from many
iterations, so the net doesn't overfit to the latest games). Each iteration does
`TRAIN_STEPS_PER_ITER` gradient steps, each on a random minibatch, minimizing:

```
loss = cross_entropy(policy_logits, MCTS_visit_distribution)   # imitate the search
     + mean_squared_error(value_pred, game_result)             # predict the winner
```

The policy head learns to imitate the search (so future *raw* policy already resembles
searched play, making the next search better), and the value head learns to predict
outcomes (so leaf evaluations get more accurate). Optimizer is AdamW with weight decay.

---

## Measuring strength (the arena)

Because there's no ground truth, each iteration plays evaluation games against two
fixed baselines and reports win/draw/loss:

- **Random** — sanity check; a learning net should crush this almost immediately.
- **Heuristic** — takes an immediate win, blocks an immediate loss, else prefers center
  columns. Beating this consistently is the meaningful "playing reasonably well" signal.

Each agent plays first in half the games to cancel first-move advantage. Note the
heuristic is deterministic, so its scores can swing all-or-nothing between iterations;
watch the **trend**, not a single line.

---

## Board-size independence

Nothing above assumes 6×7. `game.py` reads the board dimensions from `config`, and the
network's spatial dimensions follow. Change `C4_ROWS/COLS/CONNECT` and the same code
trains and plays a different game. The checkpoint stores its board and network shape so
the play GUI always reconstructs a matching network.

---

## Parallel self-play

Self-play is the dominant cost (~80% of an iteration) and is bottlenecked by
**single-threaded Python MCTS** — measured at ~60 positions/sec on one thread, with
the GPU mostly idle. `parallel_selfplay.SelfPlayPool` breaks that ceiling by running
several worker processes at once (`C4_WORKERS`, default ≈ CPU cores − 1, capped at 8):

- Each worker is a **persistent spawned process** holding its own network copy. The
  trainer broadcasts fresh CPU weights each iteration; workers `load_state_dict`, play
  their share of the games with `play_games`, and return stacked sample arrays.
- Workers infer on the **same GPU** as training. That's fine: self-play and training are
  separate phases (no overlap), and each worker spends most of its time in Python
  tree-walking, so their brief GPU calls interleave rather than collide. The parallelism
  is really about overlapping the CPU-bound Python work across cores.
- Scaling is sub-linear (the shared GPU eventually saturates and there's weight-broadcast
  overhead), but the practical speedup is a solid multiple over one thread.

## Practical performance notes

- **Do NOT enable `torch.backends.cudnn.benchmark`.** MCTS evaluates a *different batch
  size almost every step* (the count of active games / non-terminal leaves varies), so
  benchmark mode re-tunes conv kernels constantly instead of caching — measured **~10×
  slower**. Only `torch.set_float32_matmul_precision("high")` (TF32) is enabled.
- **The arena is measurement only**, so it runs every `EVAL_EVERY` iterations (default 10)
  rather than every one — recovering that time for actual training.
- **To trade strength for speed**, lower `SIMULATIONS` (weaker per-move search) or
  `GAMES_PER_ITER`. To go faster without losing quality, raise `C4_WORKERS`.
- **VRAM is not a constraint** for the net itself, but each worker holds its own CUDA
  context (~hundreds of MB), so very high `C4_WORKERS` counts add up — lower it if you hit
  GPU memory limits. The 64 GB of system RAM comfortably holds the 300k-position buffer.
- **Changing `CHANNELS`/`BLOCKS`** changes the network architecture, so an old checkpoint
  can't be loaded into the new shape — start a fresh checkpoint when you resize the net.
