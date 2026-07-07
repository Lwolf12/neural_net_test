"""Connect 4 rules on a numpy board. Pure logic, no torch.

A board is an (N_ROWS, N_COLS) int8 array: 0 empty, +1 / -1 for the two
players. Row 0 is the top; pieces fall to the lowest empty cell in a column.
`to_play` is +1 or -1 (the side about to move). The network always sees the
board from the perspective of the side to move, so it only has to learn one
"my pieces vs their pieces" view rather than a fixed X/O view.
"""

import numpy as np

from .config import N_ROWS, N_COLS, N_CONNECT

_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def new_board():
    return np.zeros((N_ROWS, N_COLS), dtype=np.int8)


def legal_mask(board):
    """Bool array of length N_COLS: True where a piece can still be dropped."""
    return board[0] == 0


def legal_actions(board):
    return np.nonzero(board[0] == 0)[0]


def _landing_row(board, col):
    """Lowest empty row in a column (pieces stack from the bottom)."""
    empty_rows = np.nonzero(board[:, col] == 0)[0]
    return int(empty_rows[-1])


def apply_move(board, col, player):
    """Return (new_board, landing_row) with `player` dropped into `col`."""
    board = board.copy()
    row = _landing_row(board, col)
    board[row, col] = player
    return board, row


def is_win_at(board, row, col):
    """Did the piece at (row, col) complete a line of N_CONNECT?

    Only the last-played cell can create a new win, so we scan the four
    directions through that cell — far cheaper than scanning the whole board.
    """
    player = board[row, col]
    if player == 0:
        return False

    for dr, dc in _DIRECTIONS:
        count = 1
        for step in (1, -1):
            rr, cc = row + dr * step, col + dc * step
            while 0 <= rr < N_ROWS and 0 <= cc < N_COLS and board[rr, cc] == player:
                count += 1
                rr += dr * step
                cc += dc * step
        if count >= N_CONNECT:
            return True
    return False


def is_full(board):
    return not bool((board[0] == 0).any())


def encode(board, to_play):
    """Two-plane network input from the perspective of `to_play`.

    plane 0 = my pieces, plane 1 = opponent pieces. Shape (2, N_ROWS, N_COLS).
    """
    mine = (board == to_play).astype(np.float32)
    theirs = (board == -to_play).astype(np.float32)
    return np.stack([mine, theirs])


def render(board):
    """Human-readable board string (handy for debugging / a future play UI)."""
    symbols = {0: ".", 1: "X", -1: "O"}
    lines = [" ".join(symbols[int(v)] for v in board[r]) for r in range(N_ROWS)]
    lines.append(" ".join(str(c) for c in range(N_COLS)))
    return "\n".join(lines)
