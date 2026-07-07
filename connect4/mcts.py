"""Batched Monte-Carlo Tree Search with a neural network guide (PUCT).

The key to using the GPU well on a small game is to search many games at once:
every game keeps its own tree, but on each simulation step we collect one leaf
per game and evaluate them all in a single batched forward pass. That turns
thousands of tiny net calls into a handful of big ones.

Value convention: every node stores stats from the perspective of the player to
move at that node. Backups therefore flip sign at each level up the tree.
"""

import math
import time

import numpy as np
import torch

from .config import C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPS, DEVICE, N_COLS
from .game import apply_move, encode, is_full, is_win_at, legal_mask


class Node:
    __slots__ = ("board", "to_play", "prior", "N", "W",
                 "children", "expanded", "is_terminal", "terminal_value")

    def __init__(self, board, to_play):
        self.board = board
        self.to_play = to_play
        self.prior = 0.0
        self.N = 0
        self.W = 0.0
        self.children = {}          # action -> Node
        self.expanded = False
        self.is_terminal = False
        self.terminal_value = 0.0   # from this node's perspective

    def q(self):
        return self.W / self.N if self.N > 0 else 0.0


@torch.no_grad()
def _evaluate(net, boards, to_plays):
    """Batched net eval. Returns (priors[G, N_COLS], values[G])."""
    x = np.stack([encode(b, p) for b, p in zip(boards, to_plays)])
    x = torch.from_numpy(x).to(DEVICE)
    logits, values = net(x)
    priors = torch.softmax(logits, dim=1).cpu().numpy()
    return priors, values.cpu().numpy()


def _expand(node, priors):
    """Create children for every legal move, seeded with masked net priors."""
    mask = legal_mask(node.board)
    p = priors * mask
    total = p.sum()
    p = p / total if total > 0 else mask / mask.sum()

    for col in np.nonzero(mask)[0]:
        child_board, row = apply_move(node.board, col, node.to_play)
        child = Node(child_board, -node.to_play)
        child.prior = float(p[col])

        if is_win_at(child_board, row, col):
            # The mover (node.to_play) just won, so the child's side-to-move lost.
            child.is_terminal = True
            child.terminal_value = -1.0
        elif is_full(child_board):
            child.is_terminal = True
            child.terminal_value = 0.0

        node.children[int(col)] = child

    node.expanded = True


def _add_dirichlet_noise(root):
    actions = list(root.children.keys())
    if not actions:
        return
    noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(actions))
    for action, eta in zip(actions, noise):
        child = root.children[action]
        child.prior = (1 - DIRICHLET_EPS) * child.prior + DIRICHLET_EPS * eta


def _select(node):
    """Pick the child maximising PUCT = Q + U (Q seen from `node`'s side)."""
    sqrt_parent = math.sqrt(max(1, node.N))
    best_score, best_action, best_child = -1e30, None, None

    for action, child in node.children.items():
        q = -child.q()  # child stats are from the opponent's perspective
        u = C_PUCT * child.prior * sqrt_parent / (1 + child.N)
        score = q + u
        if score > best_score:
            best_score, best_action, best_child = score, action, child

    return best_action, best_child


def _backup(path, value):
    """Propagate `value` (from the leaf's perspective) up the visited path."""
    sign = 1.0
    for node in reversed(path):
        node.N += 1
        node.W += sign * value
        sign = -sign


def run_mcts(net, boards, to_plays, simulations, add_noise, time_budget=None):
    """Search a batch of positions.

    Returns (visit_counts[G, N_COLS] as float32, roots) so callers can build a
    policy target from raw visit counts and, if they wish, inspect the trees.

    If `time_budget` (seconds) is given, simulations run until that wall-clock
    limit instead of a fixed count — used for play, so the AI is "as strong as
    it can be" within a chosen thinking time. `simulations` is ignored then.
    """
    roots = [Node(b, p) for b, p in zip(boards, to_plays)]

    priors, _ = _evaluate(net, boards, to_plays)
    for root, prior in zip(roots, priors):
        _expand(root, prior)
        if add_noise:
            _add_dirichlet_noise(root)

    start = time.time()
    sim = 0
    while (time.time() - start < time_budget) if time_budget else (sim < simulations):
        sim += 1
        leaf_nodes, leaf_paths = [], []

        for root in roots:
            node = root
            path = [node]
            while node.expanded and not node.is_terminal:
                _, node = _select(node)
                path.append(node)

            if node.is_terminal:
                _backup(path, node.terminal_value)
            else:
                leaf_nodes.append(node)
                leaf_paths.append(path)

        if leaf_nodes:
            lb = [n.board for n in leaf_nodes]
            lp = [n.to_play for n in leaf_nodes]
            leaf_priors, leaf_values = _evaluate(net, lb, lp)
            for node, path, pr, val in zip(leaf_nodes, leaf_paths, leaf_priors, leaf_values):
                _expand(node, pr)
                _backup(path, float(val))

    visit_counts = np.zeros((len(roots), N_COLS), dtype=np.float32)
    for i, root in enumerate(roots):
        for action, child in root.children.items():
            visit_counts[i, action] = child.N

    return visit_counts, roots
