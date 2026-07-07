"""Parallel self-play across multiple processes.

Self-play is bottlenecked by single-threaded Python MCTS, so the way to go
faster on a multi-core CPU is to run several self-play processes at once. Each
worker is a persistent spawned process that holds its own copy of the network;
every iteration the trainer broadcasts fresh weights, each worker plays its
share of the games, and returns the resulting training samples.

Workers run inference on the same GPU as training. That's fine because the two
phases don't overlap (the trainer generates games, *then* trains), and each
worker spends most of its time in Python tree-walking, so their brief GPU calls
interleave rather than collide.

Samples are shipped back as stacked numpy arrays (not per-position tuples) to
keep inter-process pickling cheap.
"""

import multiprocessing as mp

import numpy as np
import torch

from . import config as cfg
from .model import Connect4Net
from .selfplay import play_games

_EMPTY_STATE = (2, cfg.N_ROWS, cfg.N_COLS)


def _stack(samples):
    if not samples:
        return (np.empty(( 0,) + _EMPTY_STATE, np.float32),
                np.empty((0, cfg.N_COLS), np.float32),
                np.empty((0,), np.float32))
    states = np.stack([s for s, _, _ in samples])
    policies = np.stack([p for _, p, _ in samples])
    values = np.array([v for _, _, v in samples], dtype=np.float32)
    return states, policies, values


def _worker(worker_id, cmd_q, result_q):
    """Runs in a spawned process: build the net once, then serve tasks."""
    torch.set_num_threads(1)  # avoid CPU oversubscription across many workers
    net = Connect4Net().to(cfg.DEVICE)
    net.eval()

    while True:
        task = cmd_q.get()
        if task is None:
            break
        state_dict, n_games, simulations, seed = task
        net.load_state_dict(state_dict)
        np.random.seed(seed)
        torch.manual_seed(seed)
        samples, winners = play_games(net, n_games, simulations)
        result_q.put((worker_id, *_stack(samples), winners))


def _split(total, n):
    """Divide `total` games as evenly as possible into `n` positive chunks."""
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


class SelfPlayPool:
    """A persistent pool of self-play worker processes."""

    def __init__(self, n_workers):
        ctx = mp.get_context("spawn")
        self.n_workers = n_workers
        self.result_q = ctx.Queue()
        self.cmd_qs = []
        self.procs = []
        for wid in range(n_workers):
            cq = ctx.Queue()
            proc = ctx.Process(target=_worker, args=(wid, cq, self.result_q),
                               daemon=True)
            proc.start()
            self.cmd_qs.append(cq)
            self.procs.append(proc)

    def generate(self, state_dict, total_games, simulations, base_seed):
        """Broadcast weights, play `total_games` across the workers, collect samples.

        Returns (samples, winners) with the same shape the in-process
        `play_games` returns: samples is a list of (state, policy, value).
        """
        counts = _split(total_games, self.n_workers)
        for wid, cq in enumerate(self.cmd_qs):
            cq.put((state_dict, counts[wid], simulations, base_seed + wid))

        states, policies, values, winners = [], [], [], []
        for _ in range(self.n_workers):
            _wid, s, p, v, w = self.result_q.get()
            states.append(s)
            policies.append(p)
            values.append(v)
            winners.extend(w)

        states = np.concatenate(states)
        policies = np.concatenate(policies)
        values = np.concatenate(values)
        samples = list(zip(states, policies, values))
        return samples, winners

    def close(self):
        for cq in self.cmd_qs:
            cq.put(None)
        for proc in self.procs:
            proc.join(timeout=3)
            if proc.is_alive():
                proc.terminate()
