"""Self-play data generation and strength evaluation (arena).

All games in a batch advance one ply at a time in lockstep, so at any moment
every still-running game shares the same side to move. That lets us evaluate
every game's search with a single batched MCTS call.
"""

import numpy as np

from .config import N_COLS, TEMP_MOVES
from .game import apply_move, encode, is_full, is_win_at, legal_mask, new_board
from .mcts import run_mcts


def play_games(net, n_games, simulations):
    """Play `n_games` self-play games at once.

    Returns (samples, winners) where each sample is
    (encoded_state, policy_target, value_target) and winners[i] is +1 / -1 / 0.
    """
    boards = [new_board() for _ in range(n_games)]
    to_play = 1
    done = [False] * n_games
    winners = [0] * n_games
    # Per game: list of (encoded_state, policy_target, side_to_move_at_state).
    history = [[] for _ in range(n_games)]

    ply = 0
    active = list(range(n_games))
    while active:
        counts, _ = run_mcts(
            net,
            [boards[i] for i in active],
            [to_play] * len(active),
            simulations,
            add_noise=True,
        )

        for local, i in enumerate(active):
            visits = counts[local]
            total = visits.sum()
            policy = (visits / total).astype(np.float32) if total > 0 \
                else legal_mask(boards[i]).astype(np.float32)

            history[i].append((encode(boards[i], to_play), policy, to_play))

            if ply < TEMP_MOVES and total > 0:
                action = int(np.random.choice(N_COLS, p=visits / total))
            else:
                action = int(visits.argmax())

            boards[i], row = apply_move(boards[i], action, to_play)
            if is_win_at(boards[i], row, action):
                winners[i], done[i] = to_play, True
            elif is_full(boards[i]):
                winners[i], done[i] = 0, True

        to_play = -to_play
        ply += 1
        active = [i for i in range(n_games) if not done[i]]

    samples = []
    for i in range(n_games):
        for state, policy, side in history[i]:
            value = np.float32(winners[i] * side)  # +1 win / -1 loss / 0 draw for `side`
            samples.append((state, policy, value))

    return samples, winners


# --------------------------------------------------------------------------
# Arena: pit two agents against each other to measure strength.
# An "agent" is a callable taking a list of (board, to_play) and returning a
# list of chosen columns (batched so the net player uses one MCTS call).
# --------------------------------------------------------------------------

def net_agent(net, simulations):
    def choose(items):
        boards = [b for b, _ in items]
        to_plays = [p for _, p in items]
        counts, _ = run_mcts(net, boards, to_plays, simulations, add_noise=False)
        return [int(c.argmax()) for c in counts]
    return choose


def random_agent(items):
    moves = []
    for board, _ in items:
        legal = np.nonzero(legal_mask(board))[0]
        moves.append(int(np.random.choice(legal)))
    return moves


def heuristic_agent(items):
    """1-ply baseline: win now if possible, else block an immediate loss, else
    prefer central columns. A much tougher yardstick than a random player."""
    moves = []
    for board, to_play in items:
        legal = list(np.nonzero(legal_mask(board))[0])

        winning = _immediate_win(board, to_play, legal)
        if winning is not None:
            moves.append(winning)
            continue

        blocking = _immediate_win(board, -to_play, legal)
        if blocking is not None:
            moves.append(blocking)
            continue

        center = N_COLS // 2
        moves.append(min(legal, key=lambda c: abs(c - center)))
    return moves


def _immediate_win(board, player, legal):
    for col in legal:
        nb, row = apply_move(board, col, player)
        if is_win_at(nb, row, col):
            return int(col)
    return None


def arena(agent_a, agent_b, n_games):
    """Play agent_a vs agent_b; each takes first move in half the games.
    Returns (wins, draws, losses) from agent_a's perspective."""
    boards = [new_board() for _ in range(n_games)]
    a_side = [1 if i < n_games // 2 else -1 for i in range(n_games)]
    winners = [0] * n_games
    done = [False] * n_games

    to_play = 1
    while not all(done):
        active = [i for i in range(n_games) if not done[i]]
        a_games = [i for i in active if a_side[i] == to_play]
        b_games = [i for i in active if a_side[i] != to_play]

        chosen = {}
        if a_games:
            for i, mv in zip(a_games, agent_a([(boards[i], to_play) for i in a_games])):
                chosen[i] = mv
        if b_games:
            for i, mv in zip(b_games, agent_b([(boards[i], to_play) for i in b_games])):
                chosen[i] = mv

        for i in active:
            boards[i], row = apply_move(boards[i], chosen[i], to_play)
            if is_win_at(boards[i], row, chosen[i]):
                winners[i], done[i] = to_play, True
            elif is_full(boards[i]):
                winners[i], done[i] = 0, True

        to_play = -to_play

    wins = sum(winners[i] == a_side[i] for i in range(n_games))
    losses = sum(winners[i] == -a_side[i] for i in range(n_games))
    return wins, n_games - wins - losses, losses
