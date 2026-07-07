"""Tkinter Connect 4 — play Human/Computer in any combination.

Run with the project's venv Python:

    .venv\\Scripts\\python.exe play_gui.py

A setup screen picks who plays each side (human or the trained AI) plus a few
options; then a classic blue-board game screen lets a human click columns to
drop discs. The AI loads its board size and network shape from the checkpoint,
so it always matches how it was trained. The winner's line is highlighted and a
new game can be started at any time.
"""

import json
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Engine modules (connect4.*) are imported lazily on first game start, AFTER the
# board size / architecture from the checkpoint have been pushed into the
# environment — see EngineHolder. Nothing here imports torch at startup.

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(REPO_DIR, "connect4_az.pt")
SETTINGS_FILE = os.path.join(REPO_DIR, ".play_settings.json")

# Out-of-the-box: you (Red, first move) vs this project's AI playing its best
# within ~2 s of thinking. Press Enter on the setup screen to start immediately.
DEFAULTS = {
    "p1": "human",
    "p2": "computer",
    "checkpoint": DEFAULT_CHECKPOINT,
    "think_time": 2.0,      # seconds the AI searches per move (strength via time)
    "delay": 0,             # extra pause before the AI's move lands (ms)
    "device": "auto",
    "rows": 6,
    "cols": 7,
    "connect": 4,
}


def _read_store():
    try:
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _write_store(data):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def load_settings():
    """Last-used settings, falling back to DEFAULTS for anything missing."""
    saved = _read_store()
    return {k: saved.get(k, DEFAULTS[k]) for k in DEFAULTS}


def save_settings(settings):
    # Read-modify-write so unrelated keys (e.g. the window position) survive.
    data = _read_store()
    for k in DEFAULTS:
        data[k] = settings[k]
    _write_store(data)


def load_window():
    """Saved window position string ("+x+y"), or None."""
    return _read_store().get("window")


def save_window(position):
    data = _read_store()
    data["window"] = position
    _write_store(data)

# Palette.
BG = "#1b1b1f"
BOARD_BLUE = "#2a63c4"
BOARD_BLUE_HOVER = "#3f78d8"
HOLE = "#15151a"
RED = "#e5544b"
RED_DK = "#b23a33"
YELLOW = "#f4c542"
YELLOW_DK = "#c69a2c"
WIN_RING = "#ffffff"
TEXT = "#eaeaea"
MUTED = "#9aa0a6"

DISC = {1: (RED, RED_DK), -1: (YELLOW, YELLOW_DK)}
NAME = {1: "Red", -1: "Yellow"}


class EngineHolder:
    """Imports and caches the connect4 engine for a fixed board/architecture.

    The engine's board size and net width are baked in at import time (they come
    from config, which reads C4_* env vars). We therefore set the environment
    from the checkpoint before the first import; if a later game needs a
    *different* board/arch, the app must be restarted.
    """

    def __init__(self):
        self.loaded = False
        self.signature = None
        self.game = self.model = self.mcts = self.config = None
        self._nets = {}

    def load(self, rows, cols, connect, channels, blocks, device):
        signature = (rows, cols, connect, channels, blocks)
        if self.loaded:
            if signature != self.signature:
                raise RuntimeError(
                    "This session already initialised a "
                    f"{self.signature[0]}x{self.signature[1]} board / "
                    f"{self.signature[3]}ch net.\nRestart the app to use a "
                    "different board size or network architecture.")
            return

        os.environ["C4_ROWS"] = str(rows)
        os.environ["C4_COLS"] = str(cols)
        os.environ["C4_CONNECT"] = str(connect)
        if channels:
            os.environ["C4_CHANNELS"] = str(channels)
        if blocks:
            os.environ["C4_BLOCKS"] = str(blocks)
        if device in ("cuda", "cpu"):
            os.environ["C4_DEVICE"] = device

        from connect4 import config, game, mcts, model
        self.config, self.game, self.mcts, self.model = config, game, mcts, model
        self.signature = signature
        self.loaded = True

    def get_net(self, checkpoint, state_dict):
        """Build (once per checkpoint path) and cache an eval-mode net."""
        if checkpoint in self._nets:
            return self._nets[checkpoint]
        net = self.model.Connect4Net().to(self.config.DEVICE)
        net.load_state_dict(state_dict)
        net.eval()
        self._nets[checkpoint] = net
        return net


class SetupScreen(ttk.Frame):
    def __init__(self, master, on_start, initial):
        super().__init__(master, padding=24)
        self.on_start = on_start
        self.vars = {}

        ttk.Label(self, text="Connect 4", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, pady=(0, 4))
        ttk.Label(self, text="Press Enter to start, or adjust the options first.",
                  style="Muted.TLabel").grid(row=1, column=0, columnspan=2,
                                             pady=(0, 18))

        # Players.
        players = ttk.LabelFrame(self, text="Players", padding=12)
        players.grid(row=2, column=0, columnspan=2, sticky="ew", pady=6)
        self.vars["p1"] = tk.StringVar(value=initial["p1"])
        self.vars["p2"] = tk.StringVar(value=initial["p2"])
        self._player_row(players, 0, "● Red  (first move)", self.vars["p1"], RED)
        self._player_row(players, 1, "● Yellow", self.vars["p2"], YELLOW)

        # AI options.
        ai = ttk.LabelFrame(self, text="Computer (AI) options", padding=12)
        ai.grid(row=3, column=0, columnspan=2, sticky="ew", pady=6)
        ai.columnconfigure(1, weight=1)

        ttk.Label(ai, text="AI checkpoint").grid(row=0, column=0, sticky="w", pady=4)
        self.vars["checkpoint"] = tk.StringVar(value=initial["checkpoint"])
        ttk.Entry(ai, textvariable=self.vars["checkpoint"]).grid(
            row=0, column=1, sticky="ew", padx=6)
        ttk.Button(ai, text="Browse…", command=self._browse).grid(row=0, column=2)

        ttk.Label(ai, text="AI thinking time").grid(row=1, column=0, sticky="w", pady=4)
        self.vars["think_time"] = tk.DoubleVar(value=initial["think_time"])
        self._scale(ai, 1, self.vars["think_time"], 0.2, 10.0, "{:.1f} s")

        ttk.Label(ai, text="Move delay").grid(row=2, column=0, sticky="w", pady=4)
        self.vars["delay"] = tk.DoubleVar(value=initial["delay"])
        self._scale(ai, 2, self.vars["delay"], 0, 1500, "{:.0f} ms")

        ttk.Label(ai, text="Device").grid(row=3, column=0, sticky="w", pady=4)
        self.vars["device"] = tk.StringVar(value=initial["device"])
        ttk.Combobox(ai, textvariable=self.vars["device"], width=10, state="readonly",
                     values=["auto", "cuda", "cpu"]).grid(row=3, column=1, sticky="w", padx=6)

        # Board size (human-vs-human only; AI games use the checkpoint's board).
        board = ttk.LabelFrame(self, text="Board size (used only for Human vs Human)",
                               padding=12)
        board.grid(row=4, column=0, columnspan=2, sticky="ew", pady=6)
        for i, (lbl, key) in enumerate([("Rows", "rows"), ("Cols", "cols"),
                                        ("Connect", "connect")]):
            self.vars[key] = tk.IntVar(value=initial[key])
            ttk.Label(board, text=lbl).grid(row=0, column=i * 2, sticky="e", padx=(12, 4))
            ttk.Spinbox(board, from_=4, to=12, width=5,
                        textvariable=self.vars[key]).grid(row=0, column=i * 2 + 1, sticky="w")

        # Buttons.
        buttons = ttk.Frame(self)
        buttons.grid(row=5, column=0, columnspan=2, pady=(18, 0), sticky="ew")
        buttons.columnconfigure(0, weight=1)
        self.start_btn = ttk.Button(buttons, text="▶  Start game",
                                    style="Accent.TButton", command=self._start)
        self.start_btn.grid(row=0, column=0, sticky="ew")
        ttk.Button(buttons, text="Defaults", command=self._reset).grid(
            row=0, column=1, padx=(8, 0))

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.start_btn.focus_set()

    def _player_row(self, parent, row, label, var, color):
        ttk.Label(parent, text=label, foreground=color,
                  font=("Segoe UI", 11, "bold")).grid(row=row, column=0, sticky="w", padx=(2, 20))
        ttk.Radiobutton(parent, text="Human", value="human", variable=var).grid(
            row=row, column=1, padx=6)
        ttk.Radiobutton(parent, text="Computer", value="computer", variable=var).grid(
            row=row, column=2, padx=6)

    def _scale(self, parent, row, var, lo, hi, fmt):
        wrap = ttk.Frame(parent)
        wrap.grid(row=row, column=1, columnspan=2, sticky="ew", padx=6)
        wrap.columnconfigure(0, weight=1)
        ttk.Scale(wrap, from_=lo, to=hi, orient="horizontal",
                  variable=var).grid(row=0, column=0, sticky="ew")
        display = tk.StringVar()
        ttk.Label(wrap, textvariable=display, width=7).grid(row=0, column=1, padx=(8, 0))
        var.trace_add("write", lambda *_: display.set(fmt.format(var.get())))
        display.set(fmt.format(var.get()))

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select AI checkpoint", initialdir=REPO_DIR,
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")])
        if path:
            self.vars["checkpoint"].set(path)

    def _reset(self):
        for key, value in DEFAULTS.items():
            self.vars[key].set(value)

    def get_settings(self):
        return {
            "p1": self.vars["p1"].get(),
            "p2": self.vars["p2"].get(),
            "checkpoint": self.vars["checkpoint"].get().strip(),
            "think_time": round(float(self.vars["think_time"].get()), 1),
            "delay": int(self.vars["delay"].get()),
            "device": self.vars["device"].get(),
            "rows": int(self.vars["rows"].get()),
            "cols": int(self.vars["cols"].get()),
            "connect": int(self.vars["connect"].get()),
        }

    def _start(self):
        self.on_start(self.get_settings())


class GameScreen(ttk.Frame):
    CELL = 78
    MARGIN = 16
    TOP = 92

    def __init__(self, master, engine, settings, net, board_dims, on_new, on_setup):
        super().__init__(master, padding=12)
        self.engine = engine
        self.game = engine.game
        self.settings = settings
        self.net = net
        self.rows, self.cols, self.connect = board_dims
        self.on_new = on_new
        self.on_setup = on_setup

        self.ptype = {1: settings["p1"], -1: settings["p2"]}
        # The game is a line of moves (columns) plus a cursor into it. Take-back
        # moves the cursor back; forward moves it toward the line's tip; a new
        # move truncates any "future" beyond the cursor. The board / to_play /
        # winner are always derived from line[:cursor] by _apply_cursor().
        self.line = []
        self.cursor = 0
        self.board = self.game.new_board()
        self.to_play = 1
        self.winner = None
        self.busy = False
        self.game_over = False
        self.hover_col = None
        self.win_cells = set()
        self.result_q = queue.Queue()

        # Header.
        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 8))
        self.status = tk.StringVar(value="")
        ttk.Label(header, textvariable=self.status,
                  font=("Segoe UI", 15, "bold")).pack(side="left")

        # Board canvas.
        cw = self.cols * self.CELL + 2 * self.MARGIN
        ch = self.TOP + self.rows * self.CELL + self.MARGIN
        self.canvas = tk.Canvas(self, width=cw, height=ch, bg=BG,
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda e: self._set_hover(None))
        self.canvas.bind("<Button-1>", self._on_click)

        # Controls.
        controls = ttk.Frame(self)
        controls.pack(fill="x", pady=(10, 0))
        ttk.Button(controls, text="↻  New game", style="Accent.TButton",
                   command=self._confirm_new).pack(side="left")
        self.takeback_btn = ttk.Button(controls, text="↶  Take back",
                                        command=self.take_back)
        self.takeback_btn.pack(side="left", padx=(8, 2))
        self.forward_btn = ttk.Button(controls, text="Forward  ↷",
                                      command=self.redo)
        self.forward_btn.pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="⚙  Setup",
                   command=self._confirm_setup).pack(side="left")
        self.subtitle = tk.StringVar()
        ttk.Label(controls, textvariable=self.subtitle, style="Muted.TLabel").pack(side="right")

        self._describe_matchup()
        self.after(120, self._poll_ai)
        self._apply_cursor()
        self._render_after_move()

    # ---- matchup / status -------------------------------------------
    def _describe_matchup(self):
        def who(p):
            return "Computer" if self.ptype[p] == "computer" else "Human"
        self.subtitle.set(f"Red: {who(1)}   ·   Yellow: {who(-1)}   ·   "
                          f"{self.rows}×{self.cols} connect {self.connect}   ·   "
                          f"←/→: take back / forward")

    # ---- navigation (undo / redo over the move line) -----------------
    def _apply_cursor(self):
        """Rebuild board / to_play / winner by replaying line[:cursor]."""
        board = self.game.new_board()
        last = None
        for i in range(self.cursor):
            player = 1 if i % 2 == 0 else -1
            board, row = self.game.apply_move(board, self.line[i], player)
            last = (row, self.line[i], player)

        self.board = board
        self.to_play = 1 if self.cursor % 2 == 0 else -1
        self.game_over = False
        self.winner = None
        self.win_cells = set()
        if last and self.game.is_win_at(board, last[0], last[1]):
            self.game_over = True
            self.winner = last[2]
            self.win_cells = self._winning_cells(last[0], last[1], last[2])
        elif self.game.is_full(board):
            self.game_over = True  # draw (winner stays None)

    def _render_after_move(self):
        """Redraw and either declare the result or hand off to the next player."""
        self.busy = False
        self.hover_col = None
        self._draw()
        if self.game_over:
            if self.winner is None:
                text, color = "Draw — board full", MUTED
            else:
                text, color = f"★  {NAME[self.winner]} wins!  ★", DISC[self.winner][0]
            self.status.set(text)
            self._status_color(color)
            self._banner(text, color)
            self._update_nav()
        else:
            self._advance()

    def _advance(self):
        """Decide what happens next after the position changed."""
        color = DISC[self.to_play][0]
        name = NAME[self.to_play]
        if self.ptype[self.to_play] == "computer":
            self.status.set(f"{name} (Computer) is thinking…")
            self._status_color(color)
            self._start_ai()
        else:
            self.status.set(f"{name}'s turn — click a column")
            self._status_color(color)
        self._update_nav()

    def _update_nav(self):
        undo = (not self.busy) and self.cursor > 0
        redo = (not self.busy) and self.cursor < len(self.line)
        self.takeback_btn.config(state=("normal" if undo else "disabled"))
        self.forward_btn.config(state=("normal" if redo else "disabled"))

    def take_back(self):
        """Rewind to the previous position where a human is to move.

        In Human-vs-Computer this undoes the AI's reply and your own move in one
        press; repeat to go back further, all the way to the start. In
        Human-vs-Human it undoes a single ply.
        """
        if self.busy or self.cursor == 0:
            return
        human_exists = "human" in (self.ptype[1], self.ptype[-1])
        c = self.cursor
        while c > 0:
            c -= 1
            side = 1 if c % 2 == 0 else -1
            if not human_exists or self.ptype[side] == "human":
                break
        self.cursor = c
        self._apply_cursor()
        self._render_after_move()

    def redo(self):
        """Move forward again over moves that were taken back, up to the
        original position (the line's tip). Mirrors take_back."""
        if self.busy or self.cursor >= len(self.line):
            return
        human_exists = "human" in (self.ptype[1], self.ptype[-1])
        c, n = self.cursor, len(self.line)
        while c < n:
            c += 1
            if c == n:
                break  # reached the tip (which may be a finished game)
            side = 1 if c % 2 == 0 else -1
            if not human_exists or self.ptype[side] == "human":
                break
        self.cursor = c
        self._apply_cursor()
        self._render_after_move()

    # ---- cancel-guarding the New game / Setup buttons ----------------
    def _game_in_progress(self):
        return self.cursor > 0 and not self.game_over

    def _confirm_new(self):
        if self._game_in_progress() and not messagebox.askyesno(
                "Cancel current game?",
                "A game is in progress — starting a new game will end it.\n\n"
                "Start a new game?"):
            return
        self.on_new()

    def _confirm_setup(self):
        if self._game_in_progress() and not messagebox.askyesno(
                "Cancel current game?",
                "A game is in progress — returning to setup will end it.\n\n"
                "Return to setup?"):
            return
        self.on_setup()

    def _status_color(self, color):
        # Recolour the status label to the side to move.
        for child in self.winfo_children():
            if isinstance(child, ttk.Frame):
                for w in child.winfo_children():
                    if isinstance(w, ttk.Label) and str(w.cget("textvariable")) == str(self.status):
                        w.configure(foreground=color)

    # ---- AI ----------------------------------------------------------
    def _start_ai(self):
        self.busy = True
        board = self.board.copy()
        to_play = self.to_play
        think_time = self.settings["think_time"]

        def work():
            counts, _ = self.engine.mcts.run_mcts(
                self.net, [board], [to_play], 0,
                add_noise=False, time_budget=think_time)
            self.result_q.put(int(counts[0].argmax()))

        threading.Thread(target=work, daemon=True).start()

    def _poll_ai(self):
        try:
            action = self.result_q.get_nowait()
        except queue.Empty:
            action = None
        if action is not None and not self.game_over:
            self.after(self.settings["delay"], lambda a=action: self._play(a))
        self.after(120, self._poll_ai)

    # ---- human input -------------------------------------------------
    def _on_motion(self, event):
        if self.busy or self.game_over or self.ptype[self.to_play] == "computer":
            self._set_hover(None)
            return
        col = self._col_at(event.x)
        self._set_hover(col if col is not None and self._playable(col) else None)

    def _on_click(self, event):
        if self.busy or self.game_over or self.ptype[self.to_play] == "computer":
            return
        col = self._col_at(event.x)
        if col is not None and self._playable(col):
            self._play(col)

    def _col_at(self, x):
        c = int((x - self.MARGIN) // self.CELL)
        return c if 0 <= c < self.cols else None

    def _playable(self, col):
        return bool(self.game.legal_mask(self.board)[col])

    def _set_hover(self, col):
        if col != self.hover_col:
            self.hover_col = col
            self._draw()

    # ---- move + animation -------------------------------------------
    def _play(self, col):
        if not self._playable(col) or self.game_over:
            return
        self.busy = True
        self.hover_col = None
        player = self.to_play
        row = self.game._landing_row(self.board, col)
        self._animate(col, row, player, lambda: self._land(col))

    def _land(self, col):
        # Commit the move: any taken-back "future" is discarded, then advance.
        self.line = self.line[:self.cursor]
        self.line.append(col)
        self.cursor += 1
        self._apply_cursor()
        self._render_after_move()

    def _animate(self, col, row, player, on_done):
        cx = self.MARGIN + col * self.CELL + self.CELL / 2
        target = self.TOP + row * self.CELL + self.CELL / 2
        r = self.CELL * 0.40
        fill, outline = DISC[player]

        self._draw()  # static board without the new disc
        disc = self.canvas.create_oval(cx - r, self.TOP / 2 - r, cx + r,
                                       self.TOP / 2 + r, fill=fill,
                                       outline=outline, width=3)
        state = {"y": self.TOP / 2, "v": 0.0}
        gravity = self.CELL * 0.10

        def step():
            state["v"] += gravity
            state["y"] += state["v"]
            if state["y"] >= target:
                state["y"] = target
                self.canvas.coords(disc, cx - r, target - r, cx + r, target + r)
                self.canvas.delete(disc)
                on_done()
                return
            self.canvas.coords(disc, cx - r, state["y"] - r, cx + r, state["y"] + r)
            self.after(12, step)

        step()

    # ---- drawing -----------------------------------------------------
    def _draw(self):
        c = self.canvas
        c.delete("all")
        left, top = self.MARGIN, self.TOP
        right = left + self.cols * self.CELL
        bottom = top + self.rows * self.CELL

        # Hover column highlight + floating preview disc.
        if self.hover_col is not None:
            hx = left + self.hover_col * self.CELL
            c.create_rectangle(hx, top, hx + self.CELL, bottom,
                               fill=BOARD_BLUE_HOVER, outline="")
            pcx = hx + self.CELL / 2
            pr = self.CELL * 0.40
            fill, outline = DISC[self.to_play]
            c.create_oval(pcx - pr, self.TOP / 2 - pr, pcx + pr, self.TOP / 2 + pr,
                          fill=fill, outline=outline, width=3)

        # Blue board (skip the hovered column, already painted lighter).
        for col in range(self.cols):
            if col == self.hover_col:
                continue
            x = left + col * self.CELL
            c.create_rectangle(x, top, x + self.CELL, bottom,
                               fill=BOARD_BLUE, outline="")

        # Holes + settled discs.
        for r in range(self.rows):
            for col in range(self.cols):
                cx = left + col * self.CELL + self.CELL / 2
                cy = top + r * self.CELL + self.CELL / 2
                rad = self.CELL * 0.40
                val = int(self.board[r, col])
                if val == 0:
                    c.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                                  fill=HOLE, outline="")
                else:
                    fill, outline = DISC[val]
                    c.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                                  fill=fill, outline=outline, width=3)
                    if (r, col) in self.win_cells:
                        c.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                                      outline=WIN_RING, width=5)

    def _banner(self, text, color):
        c = self.canvas
        w = int(c["width"])
        cx = w / 2
        cy = self.TOP / 2
        c.create_rectangle(cx - 190, cy - 26, cx + 190, cy + 26,
                           fill="#000000", outline=color, width=3)
        c.create_text(cx, cy, text=text, fill=color,
                      font=("Segoe UI", 18, "bold"))

    def _winning_cells(self, row, col, player):
        cells = {(row, col)}
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            line = [(row, col)]
            for step in (1, -1):
                rr, cc = row + dr * step, col + dc * step
                while (0 <= rr < self.rows and 0 <= cc < self.cols
                       and int(self.board[rr, cc]) == player):
                    line.append((rr, cc))
                    rr += dr * step
                    cc += dc * step
            if len(line) >= self.connect:
                cells.update(line)
        return cells


class App:
    def __init__(self, root):
        self.root = root
        root.title("Connect 4")
        root.configure(bg=BG)
        self.engine = EngineHolder()
        self.settings = load_settings()
        self.current = None

        self._style()
        self._restore_window()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.bind("<Return>", self._on_return)
        for seq in ("<Left>", "<BackSpace>", "<Control-z>"):
            root.bind(seq, self._on_takeback)
        for seq in ("<Right>", "<Control-y>"):
            root.bind(seq, self._on_forward)
        self._show_setup()

    # ---- window position memory --------------------------------------
    def _restore_window(self):
        pos = load_window()
        if pos:
            try:
                self.root.geometry(pos)  # position only ("+x+y"); size is content-driven
            except tk.TclError:
                pass

    def _on_close(self):
        try:
            geom = self.root.geometry()  # "WxH+X+Y"
            idx = next((i for i, ch in enumerate(geom) if ch in "+-"), -1)
            if idx > 0:
                save_window(geom[idx:])
        except tk.TclError:
            pass
        self.root.destroy()

    def _on_return(self, _event):
        # Enter on the setup screen starts the game immediately.
        if isinstance(self.current, SetupScreen):
            self.current._start()

    def _on_takeback(self, _event):
        # ←/Backspace/Ctrl+Z take back moves during a game (ignored elsewhere,
        # so text entries on the setup screen keep their normal behaviour).
        if isinstance(self.current, GameScreen):
            self.current.take_back()

    def _on_forward(self, _event):
        # →/Ctrl+Y move forward again over taken-back moves.
        if isinstance(self.current, GameScreen):
            self.current.redo()

    def _style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG, foreground=TEXT)
        style.configure("TFrame", background=BG)
        style.configure("TLabelframe", background=BG, foreground=TEXT)
        style.configure("TLabelframe.Label", background=BG, foreground="#7fb0ff")
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Muted.TLabel", background=BG, foreground=MUTED)
        style.configure("Title.TLabel", background=BG, foreground=TEXT,
                        font=("Segoe UI", 26, "bold"))
        style.configure("TButton", padding=6)
        style.configure("Accent.TButton", padding=8, font=("Segoe UI", 11, "bold"))
        style.configure("TRadiobutton", background=BG, foreground=TEXT)
        style.map("TRadiobutton", background=[("active", BG)])

        # Input fields need an explicit dark field background, otherwise the
        # light global foreground would be invisible on clam's default white.
        field_bg = "#33333a"
        for widget in ("TEntry", "TSpinbox", "TCombobox"):
            style.configure(widget, fieldbackground=field_bg, foreground=TEXT,
                            insertcolor=TEXT, arrowcolor=TEXT)
        style.map("TCombobox",
                  fieldbackground=[("readonly", field_bg)],
                  foreground=[("readonly", TEXT)])
        # The combobox dropdown list is a classic Tk widget, not ttk.
        self.root.option_add("*TCombobox*Listbox.background", field_bg)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)

    def _clear(self):
        if self.current is not None:
            self.current.destroy()
            self.current = None

    def _show_setup(self):
        self._clear()
        self.current = SetupScreen(self.root, self._start_game, self.settings)
        self.current.pack(fill="both", expand=True)

    def _start_game(self, settings):
        need_ai = "computer" in (settings["p1"], settings["p2"])
        net = None
        board_dims = (settings["rows"], settings["cols"], settings["connect"])

        try:
            if need_ai:
                import torch
                path = settings["checkpoint"]
                if not os.path.isfile(path):
                    messagebox.showerror("Checkpoint not found",
                                         f"No AI checkpoint at:\n{path}")
                    return
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                rows, cols, connect = ckpt.get("board", (6, 7, 4))
                board_dims = (rows, cols, connect)
                self.engine.load(rows, cols, connect,
                                 ckpt.get("channels"), ckpt.get("blocks"),
                                 settings["device"])
                net = self.engine.get_net(path, ckpt["model"])
            else:
                self.engine.load(*board_dims, None, None, settings["device"])
        except Exception as exc:  # surfaces mismatch / load errors to the user
            messagebox.showerror("Could not start game", str(exc))
            return

        self.settings = settings
        save_settings(settings)  # remember for next launch
        self._clear()
        self.current = GameScreen(self.root, self.engine, settings, net,
                                  board_dims, on_new=self._new_game,
                                  on_setup=self._show_setup)
        self.current.pack(fill="both", expand=True)

    def _new_game(self):
        self._start_game(self.settings)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
