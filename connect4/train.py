"""Runnable trainer:  python -m connect4.train

Loop forever (Ctrl-C to stop, progress is checkpointed every iteration):

    1. self-play      - generate games with the current net + MCTS
    2. learn          - gradient steps on a replay buffer of recent positions
    3. evaluate       - win rate vs random and vs a 1-ply heuristic baseline

The value net predicts the game result; the policy head is trained to imitate
the MCTS visit distribution (which is stronger than the raw policy). Over
iterations the two reinforce each other and play improves.
"""

import collections
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import config as cfg
from .model import Connect4Net
from .parallel_selfplay import SelfPlayPool
from .selfplay import (arena, heuristic_agent, net_agent, play_games,
                       random_agent)


def _resolve_workers():
    if cfg.WORKERS > 0:
        return cfg.WORKERS
    return min(8, max(1, (os.cpu_count() or 2) - 1))


def _train_steps(net, optimizer, buffer, steps):
    data = list(buffer)
    states = np.stack([s for s, _, _ in data])
    policies = np.stack([p for _, p, _ in data])
    values = np.array([v for _, _, v in data], dtype=np.float32)

    states = torch.from_numpy(states).to(cfg.DEVICE)
    policies = torch.from_numpy(policies).to(cfg.DEVICE)
    values = torch.from_numpy(values).to(cfg.DEVICE)

    n = len(data)
    net.train()
    total_p = total_v = 0.0
    for _ in range(steps):
        idx = torch.randint(0, n, (cfg.BATCH_SIZE,), device=cfg.DEVICE)
        logits, value_pred = net(states[idx])

        policy_loss = -(policies[idx] * F.log_softmax(logits, dim=1)).sum(1).mean()
        value_loss = F.mse_loss(value_pred, values[idx])
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_p += policy_loss.item()
        total_v += value_loss.item()

    return total_p / steps, total_v / steps


def _save(path, net, optimizer, iteration):
    torch.save(
        {
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "board": (cfg.N_ROWS, cfg.N_COLS, cfg.N_CONNECT),
            "channels": cfg.CHANNELS,
            "blocks": cfg.BLOCKS,
        },
        path,
    )


def main():
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    # Allow TF32 matmuls on Ada (negligible accuracy impact, slightly faster).
    # NOTE: deliberately NOT enabling torch.backends.cudnn.benchmark — MCTS
    # evaluates a different batch size almost every step, so benchmark mode
    # re-tunes conv kernels constantly and is ~10x SLOWER for this workload.
    if cfg.DEVICE == "cuda":
        torch.set_float32_matmul_precision("high")

    net = Connect4Net().to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    buffer = collections.deque(maxlen=cfg.REPLAY_CAPACITY)
    start = 0

    if os.path.exists(cfg.CHECKPOINT):
        ckpt = torch.load(cfg.CHECKPOINT, map_location=cfg.DEVICE)
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start = ckpt["iteration"]
        print(f"Resumed from {cfg.CHECKPOINT} at iteration {start}.")

    n_workers = _resolve_workers()
    pool = SelfPlayPool(n_workers) if n_workers > 1 else None

    print(f"Device: {cfg.DEVICE} | board: {cfg.N_ROWS}x{cfg.N_COLS} "
          f"connect {cfg.N_CONNECT} | net: {cfg.CHANNELS}ch x {cfg.BLOCKS} blocks")
    print(f"Self-play: {cfg.GAMES_PER_ITER} games x {cfg.SIMULATIONS} sims/move "
          f"across {n_workers} worker(s)\n")

    # Machine-readable line for the GUI to parse (ignored by a human reader).
    print("@@INIT@@ " + json.dumps({
        "device": cfg.DEVICE,
        "board": f"{cfg.N_ROWS}x{cfg.N_COLS} connect {cfg.N_CONNECT}",
        "net": f"{cfg.CHANNELS}ch x {cfg.BLOCKS}",
        "iterations_total": cfg.ITERATIONS,
        "games_per_iter": cfg.GAMES_PER_ITER,
        "simulations": cfg.SIMULATIONS,
        "arena_games": cfg.ARENA_GAMES,
        "workers": n_workers,
        "checkpoint": cfg.CHECKPOINT,
        "start_iteration": start,
    }), flush=True)

    run_start = time.time()

    try:
        for iteration in range(start, cfg.ITERATIONS):
            t0 = time.time()

            net.eval()
            if pool:
                weights = {k: v.detach().cpu() for k, v in net.state_dict().items()}
                samples, winners = pool.generate(
                    weights, cfg.GAMES_PER_ITER, cfg.SIMULATIONS,
                    base_seed=cfg.SEED + (iteration + 1) * n_workers)
            else:
                samples, winners = play_games(net, cfg.GAMES_PER_ITER, cfg.SIMULATIONS)
            buffer.extend(samples)

            p_loss = v_loss = float("nan")
            if len(buffer) >= cfg.BATCH_SIZE:
                p_loss, v_loss = _train_steps(
                    net, optimizer, buffer, cfg.TRAIN_STEPS_PER_ITER
                )

            # Arena is measurement only, so run it periodically, not every iter.
            do_eval = (iteration == start) or ((iteration + 1) % cfg.EVAL_EVERY == 0)
            arena_stats = None
            if do_eval:
                net.eval()
                rw, rd, rl = arena(net_agent(net, cfg.ARENA_SIMS), random_agent, cfg.ARENA_GAMES)
                hw, hd, hl = arena(net_agent(net, cfg.ARENA_SIMS), heuristic_agent, cfg.ARENA_GAMES)
                arena_stats = {
                    "vs_random": {"w": rw, "d": rd, "l": rl},
                    "vs_heuristic": {"w": hw, "d": hd, "l": hl},
                }

            first = sum(w == 1 for w in winners)
            second = sum(w == -1 for w in winners)
            draws = sum(w == 0 for w in winners)
            iter_seconds = time.time() - t0

            arena_str = (f"vs random {rw}/{rd}/{rl} | vs heuristic {hw}/{hd}/{hl}"
                         if do_eval else "arena skipped")
            print(
                f"iter {iteration + 1:4d} | {iter_seconds:5.1f}s | "
                f"buffer {len(buffer):6d} | "
                f"loss p {p_loss:.3f} v {v_loss:.3f} | "
                f"selfplay W/L/D {first}/{second}/{draws} | {arena_str}"
            )
            print("@@STATS@@ " + json.dumps({
                "iteration": iteration + 1,
                "iter_seconds": round(iter_seconds, 2),
                "elapsed_seconds": round(time.time() - run_start, 2),
                "buffer": len(buffer),
                "policy_loss": None if p_loss != p_loss else round(p_loss, 4),
                "value_loss": None if v_loss != v_loss else round(v_loss, 4),
                "selfplay": {"first": first, "second": second, "draws": draws},
                "vs_random": arena_stats["vs_random"] if arena_stats else None,
                "vs_heuristic": arena_stats["vs_heuristic"] if arena_stats else None,
            }), flush=True)

            _save(cfg.CHECKPOINT, net, optimizer, iteration + 1)

    except KeyboardInterrupt:
        _save(cfg.CHECKPOINT, net, optimizer, iteration)
        print(f"\nInterrupted. Saved checkpoint to {cfg.CHECKPOINT}.")
    finally:
        if pool:
            pool.close()


if __name__ == "__main__":
    main()
