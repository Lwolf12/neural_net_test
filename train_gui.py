"""Tkinter control panel for the Connect 4 AlphaZero trainer.

Run it with the project's venv Python:

    .venv\\Scripts\\python.exe train_gui.py

It launches `python -m connect4.train` as a subprocess, passes your chosen
parameters via C4_* environment variables, and shows live statistics parsed
from the trainer's @@INIT@@ / @@STATS@@ output lines. Start/Stop are safe: the
trainer checkpoints every iteration and resumes from that checkpoint.
"""

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk

# Pull the defaults straight from the trainer's config so the two never drift.
from connect4 import config as cfg

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# (field key, label, C4_ env var, default value as string).
# `iterations` defaults to a friendly finite run rather than config's "run
# forever" sentinel; everything else mirrors config.py.
PARAMETERS = [
    ("iterations",      "Iterations",          "C4_ITERATIONS",       "200"),
    ("games_per_iter",  "Self-play games/iter", "C4_GAMES_PER_ITER",  str(cfg.GAMES_PER_ITER)),
    ("simulations",     "MCTS sims/move",       "C4_SIMULATIONS",     str(cfg.SIMULATIONS)),
    ("workers",         "Self-play workers (0=auto)", "C4_WORKERS",   str(cfg.WORKERS)),
    ("train_steps",     "Train steps/iter",     "C4_TRAIN_STEPS",     str(cfg.TRAIN_STEPS_PER_ITER)),
    ("batch_size",      "Batch size",           "C4_BATCH_SIZE",      str(cfg.BATCH_SIZE)),
    ("lr",              "Learning rate",        "C4_LR",              str(cfg.LEARNING_RATE)),
    ("weight_decay",    "Weight decay",         "C4_WEIGHT_DECAY",    str(cfg.WEIGHT_DECAY)),
    ("replay_capacity", "Replay capacity",      "C4_REPLAY_CAPACITY", str(cfg.REPLAY_CAPACITY)),
    ("channels",        "Net channels",         "C4_CHANNELS",        str(cfg.CHANNELS)),
    ("blocks",          "Net res-blocks",       "C4_BLOCKS",          str(cfg.BLOCKS)),
    ("c_puct",          "PUCT c",               "C4_C_PUCT",          str(cfg.C_PUCT)),
    ("temp_moves",      "Temperature plies",    "C4_TEMP_MOVES",      str(cfg.TEMP_MOVES)),
    ("arena_games",     "Arena games",          "C4_ARENA_GAMES",     str(cfg.ARENA_GAMES)),
    ("arena_sims",      "Arena sims",           "C4_ARENA_SIMS",      str(cfg.ARENA_SIMS)),
    ("seed",            "Seed",                 "C4_SEED",            str(cfg.SEED)),
    ("checkpoint",      "Checkpoint file",      "C4_CHECKPOINT",      cfg.CHECKPOINT),
]


def _fmt_time(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _winrate(w, d, l):
    total = w + d + l
    return 100.0 * w / total if total else 0.0


class LinePlot(tk.Canvas):
    """A tiny dependency-free multi-series line chart."""

    def __init__(self, master, title, y_range=None, width=440, height=170):
        super().__init__(master, width=width, height=height,
                         bg="#1e1e1e", highlightthickness=0)
        self.title = title
        self.y_range = y_range          # (lo, hi) or None to autoscale
        self.series = {}                # name -> (color, [values])
        self.bind("<Configure>", lambda e: self.redraw())

    def add_series(self, name, color):
        self.series[name] = (color, [])

    def push(self, name, value):
        self.series[name][1].append(value)

    def redraw(self):
        self.delete("all")
        w = self.winfo_width() or int(self["width"])
        h = self.winfo_height() or int(self["height"])
        pad_l, pad_r, pad_t, pad_b = 42, 10, 22, 18
        x0, x1 = pad_l, w - pad_r
        y0, y1 = h - pad_b, pad_t

        self.create_text(pad_l, 4, anchor="nw", text=self.title,
                         fill="#dddddd", font=("Segoe UI", 9, "bold"))

        values = [v for _, (_, vals) in self.series.items() for v in vals]
        if not values:
            return

        if self.y_range:
            lo, hi = self.y_range
        else:
            lo, hi = min(values), max(values)
            if lo == hi:
                lo, hi = lo - 1, hi + 1
            span = hi - lo
            lo, hi = lo - span * 0.08, hi + span * 0.08

        # Axes + horizontal gridlines with labels.
        self.create_line(x0, y0, x1, y0, fill="#555555")
        self.create_line(x0, y0, x0, y1, fill="#555555")
        for frac in (0.0, 0.5, 1.0):
            gy = y0 + (y1 - y0) * frac
            val = lo + (hi - lo) * frac
            self.create_line(x0, gy, x1, gy, fill="#333333")
            self.create_text(x0 - 4, gy, anchor="e", text=f"{val:.2g}",
                             fill="#999999", font=("Segoe UI", 7))

        max_len = max(len(vals) for _, (_, vals) in self.series.items())
        legend_x = x0 + 6
        for name, (color, vals) in self.series.items():
            if len(vals) >= 2:
                pts = []
                for i, v in enumerate(vals):
                    px = x0 + (x1 - x0) * (i / (max_len - 1)) if max_len > 1 else x0
                    py = y0 + (y1 - y0) * ((v - lo) / (hi - lo))
                    pts += [px, py]
                self.create_line(*pts, fill=color, width=2, smooth=True)
            self.create_text(legend_x, y1 + 2, anchor="nw", text=name,
                             fill=color, font=("Segoe UI", 8))
            legend_x += 10 + len(name) * 7


class TrainerGUI:
    def __init__(self, root):
        self.root = root
        root.title("Connect 4 AlphaZero — Trainer")
        root.configure(bg="#252526")
        root.minsize(940, 720)

        self.proc = None
        self.queue = queue.Queue()
        self.entries = {}
        self.stat_vars = {}
        self.total_iters = None
        self.best_heur = 0.0

        self._build_style()
        self._build_params()
        self._build_controls()
        self._build_stats()
        self._build_plots()
        self._build_log()

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(120, self._poll_queue)

    # ---- layout ------------------------------------------------------
    def _build_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TLabel", background="#252526", foreground="#dddddd")
        style.configure("TLabelframe", background="#252526", foreground="#dddddd")
        style.configure("TLabelframe.Label", background="#252526", foreground="#4ec9b0")
        style.configure("TButton", padding=6)

    def _build_params(self):
        frame = ttk.LabelFrame(self.root, text="Parameters")
        frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=8)

        per_col = (len(PARAMETERS) + 1) // 2
        for i, (key, label, _env, default) in enumerate(PARAMETERS):
            col = (i // per_col) * 2
            row = i % per_col
            ttk.Label(frame, text=label).grid(row=row, column=col, sticky="w",
                                              padx=(8, 4), pady=3)
            var = tk.StringVar(value=default)
            entry = ttk.Entry(frame, textvariable=var, width=14)
            entry.grid(row=row, column=col + 1, sticky="w", padx=(0, 12), pady=3)
            self.entries[key] = var

        # Device selector.
        drow = per_col
        ttk.Label(frame, text="Device").grid(row=drow, column=0, sticky="w",
                                             padx=(8, 4), pady=3)
        self.device_var = tk.StringVar(value="auto")
        ttk.Combobox(frame, textvariable=self.device_var, width=11,
                     state="readonly",
                     values=["auto", "cuda", "cpu"]).grid(
            row=drow, column=1, sticky="w", padx=(0, 12), pady=3)

    def _build_controls(self):
        bar = ttk.Frame(self.root)
        bar.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))

        self.start_btn = ttk.Button(bar, text="▶  Start", command=self.start)
        self.start_btn.pack(side="left", padx=(0, 6))
        self.stop_btn = ttk.Button(bar, text="■  Stop", command=self.stop,
                                   state="disabled")
        self.stop_btn.pack(side="left")

        self.status_var = tk.StringVar(value="idle")
        ttk.Label(bar, textvariable=self.status_var,
                  font=("Segoe UI", 10, "bold")).pack(side="right")
        ttk.Label(bar, text="Status: ").pack(side="right")

    def _build_stats(self):
        frame = ttk.LabelFrame(self.root, text="Live statistics")
        frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=6)

        fields = [
            ("iteration", "Iteration"),
            ("elapsed", "Elapsed"),
            ("iter_time", "Last iter"),
            ("avg_time", "Avg iter"),
            ("eta", "ETA"),
            ("buffer", "Replay buffer"),
            ("policy_loss", "Policy loss"),
            ("value_loss", "Value loss"),
            ("selfplay", "Self-play W/L/D"),
            ("vs_random", "vs Random (win%)"),
            ("vs_heuristic", "vs Heuristic (win%)"),
            ("best_heur", "Best heuristic win%"),
        ]
        cols = 3
        for i, (key, label) in enumerate(fields):
            r, c = divmod(i, cols)
            cell = ttk.Frame(frame)
            cell.grid(row=r, column=c, sticky="w", padx=12, pady=5)
            ttk.Label(cell, text=label, foreground="#9cdcfe",
                      font=("Segoe UI", 8)).pack(anchor="w")
            var = tk.StringVar(value="—")
            ttk.Label(cell, textvariable=var,
                      font=("Consolas", 12, "bold")).pack(anchor="w")
            self.stat_vars[key] = var

    def _build_plots(self):
        frame = ttk.Frame(self.root)
        frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=6)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        self.winrate_plot = LinePlot(frame, "Win rate vs baselines (%)",
                                     y_range=(0, 100))
        self.winrate_plot.add_series("random", "#4ec9b0")
        self.winrate_plot.add_series("heuristic", "#dcdcaa")
        self.winrate_plot.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.loss_plot = LinePlot(frame, "Training loss")
        self.loss_plot.add_series("policy", "#569cd6")
        self.loss_plot.add_series("value", "#ce9178")
        self.loss_plot.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

    def _build_log(self):
        frame = ttk.LabelFrame(self.root, text="Trainer output")
        frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=(6, 10))
        self.root.rowconfigure(4, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.log = tk.Text(frame, height=9, bg="#1e1e1e", fg="#cccccc",
                           insertbackground="#cccccc", wrap="none",
                           font=("Consolas", 9))
        self.log.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(frame, command=self.log.yview)
        sb.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=sb.set, state="disabled")

    # ---- run control -------------------------------------------------
    def start(self):
        if self.proc is not None:
            return

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = REPO_DIR + os.pathsep + env.get("PYTHONPATH", "")
        for key, _label, envname, _default in PARAMETERS:
            value = self.entries[key].get().strip()
            if value:
                env[envname] = value
        if self.device_var.get() in ("cuda", "cpu"):
            env["C4_DEVICE"] = self.device_var.get()

        try:
            iters = int(self.entries["iterations"].get())
            self.total_iters = iters
        except ValueError:
            self.total_iters = None
        self.best_heur = 0.0

        self._append_log(f"$ {sys.executable} -m connect4.train\n")
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "connect4.train"],
            cwd=REPO_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        threading.Thread(target=self._reader, args=(self.proc,),
                         daemon=True).start()

        self.status_var.set("running")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

    def stop(self):
        if self.proc is not None:
            self.status_var.set("stopping…")
            self._kill_tree(self.proc)

    def _kill_tree(self, proc):
        # The trainer spawns self-play worker child processes; kill the whole
        # tree so none are orphaned (an abrupt terminate() skips pool cleanup).
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True)
            else:
                proc.terminate()
        except Exception:
            pass

    def _reader(self, proc):
        for line in proc.stdout:
            self.queue.put(line.rstrip("\n"))
        self.queue.put(f"@@EXIT@@ {proc.wait()}")

    # ---- event pump --------------------------------------------------
    def _poll_queue(self):
        try:
            while True:
                self._handle_line(self.queue.get_nowait())
        except queue.Empty:
            pass
        self.root.after(120, self._poll_queue)

    def _handle_line(self, line):
        if line.startswith("@@STATS@@"):
            self._update_stats(_parse_json(line))
        elif line.startswith("@@INIT@@"):
            info = _parse_json(line)
            if info.get("iterations_total"):
                # Show configured total unless it's the "run forever" sentinel.
                if info["iterations_total"] < 100_000:
                    self.total_iters = info["iterations_total"]
            self._append_log(line + "\n")
        elif line.startswith("@@EXIT@@"):
            code = line.split(" ", 1)[1] if " " in line else "?"
            self.status_var.set(f"finished (exit {code})")
            self.proc = None
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
        else:
            self._append_log(line + "\n")

    def _update_stats(self, s):
        if not s:
            return
        it = s.get("iteration")
        total = f"/{self.total_iters}" if self.total_iters else ""
        self.stat_vars["iteration"].set(f"{it}{total}")

        elapsed = s.get("elapsed_seconds", 0)
        self.stat_vars["elapsed"].set(_fmt_time(elapsed))
        self.stat_vars["iter_time"].set(f"{s.get('iter_seconds', 0):.1f}s")

        avg = elapsed / it if it else 0
        self.stat_vars["avg_time"].set(f"{avg:.1f}s")
        if self.total_iters and it:
            self.stat_vars["eta"].set(_fmt_time(avg * (self.total_iters - it)))
        else:
            self.stat_vars["eta"].set("—")

        self.stat_vars["buffer"].set(f"{s.get('buffer', 0):,}")
        self.stat_vars["policy_loss"].set(_maybe(s.get("policy_loss")))
        self.stat_vars["value_loss"].set(_maybe(s.get("value_loss")))

        sp = s.get("selfplay", {})
        self.stat_vars["selfplay"].set(
            f"{sp.get('first', 0)}/{sp.get('second', 0)}/{sp.get('draws', 0)}")

        # Arena runs only every few iterations, so these may be absent.
        r = s.get("vs_random")
        h = s.get("vs_heuristic")
        if r is not None and h is not None:
            rr = _winrate(r["w"], r["d"], r["l"])
            hr = _winrate(h["w"], h["d"], h["l"])
            self.stat_vars["vs_random"].set(f"{r['w']}/{r['d']}/{r['l']}  ({rr:.0f}%)")
            self.stat_vars["vs_heuristic"].set(f"{h['w']}/{h['d']}/{h['l']}  ({hr:.0f}%)")
            self.best_heur = max(self.best_heur, hr)
            self.stat_vars["best_heur"].set(f"{self.best_heur:.0f}%")
            self.winrate_plot.push("random", rr)
            self.winrate_plot.push("heuristic", hr)
            self.winrate_plot.redraw()

        if s.get("policy_loss") is not None:
            self.loss_plot.push("policy", s["policy_loss"])
            self.loss_plot.push("value", s.get("value_loss", 0) or 0)
            self.loss_plot.redraw()

    def _append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        # Keep the log bounded.
        if int(self.log.index("end-1c").split(".")[0]) > 500:
            self.log.delete("1.0", "200.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _on_close(self):
        if self.proc is not None:
            self._kill_tree(self.proc)
        self.root.destroy()


def _parse_json(line):
    import json
    try:
        return json.loads(line.split(" ", 1)[1])
    except (IndexError, ValueError):
        return {}


def _maybe(v):
    return "—" if v is None else f"{v:.4f}"


if __name__ == "__main__":
    root = tk.Tk()
    TrainerGUI(root)
    root.mainloop()
