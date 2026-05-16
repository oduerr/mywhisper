#!/usr/bin/env python3
"""
mywhisper — hold right-⌘ (right Command) to record, release to transcribe and paste.
Change HOTKEY below to any pynput Key if you prefer a different key.

Usage:
    python app.py
    python app.py --model mlx-community/whisper-large-v3
    python app.py --help
"""

import argparse
import json
import math
import re
import subprocess
import threading
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk
import webbrowser

import numpy as np
import pyperclip
import sounddevice as sd
import mlx_whisper
from pynput import keyboard
from pynput.keyboard import Controller as KBController, Key

# ── CLI ────────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(
    description="mywhisper — hold right-⌥ to record and paste transcription at cursor",
    formatter_class=argparse.RawTextHelpFormatter,
)
_parser.add_argument(
    "--model", default="mlx-community/whisper-small-mlx",
    metavar="HF_REPO",
    help=(
        "mlx-community HuggingFace model repo (default: whisper-small-mlx).\n"
        "Common choices:\n"
        "  mlx-community/whisper-tiny-mlx          (fastest, lowest accuracy)\n"
        "  mlx-community/whisper-small-mlx          (good multilingual, ~0.3s)\n"
        "  mlx-community/whisper-medium-mlx\n"
        "  mlx-community/whisper-large-v3-turbo     (recommended, no -mlx suffix)\n"
        "Browse all: https://huggingface.co/collections/mlx-community/whisper-663256f9964fbb1177db93dc\n"
        "Note: most MLX models require a '-mlx' suffix (e.g. whisper-small-mlx),\n"
        "except whisper-large-v3-turbo which has no suffix."
    ),
)
_parser.add_argument(
    "--translate", action="store_true", default=False,
    help="Translate speech to English (instead of transcribing in the original language).",
)
_args = _parser.parse_args()

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL       = _args.model
TASK        = "translate" if _args.translate else "transcribe"
ACTIONS_FILE = Path(__file__).with_name("actions.json")
GITHUB_URL   = "https://github.com/oduerr/mywhisper"
HOTKEY      = Key.cmd_r          # right Command — easy to hold, rarely conflicts
SAMPLE_RATE = 16000
WIN_W       = 360
WIN_H       = 72
BAR_N       = 48
FPS         = 30

# ── Palette ────────────────────────────────────────────────────────────────────
BG   = "#0f172a"
STATES = {
    #           dot        status text   bar base
    "loading":      ("#475569", "#475569", "#1e293b"),
    "idle":         ("#22c55e", "#64748b", "#1e3a5f"),
    "recording":    ("#ef4444", "#ef4444", "#7f1d1d"),
    "transcribing": ("#f59e0b", "#f59e0b", "#78350f"),
}
LABELS_BASE = {
    "loading":      "Loading model…",
    "idle":         "Ready  ·  hold right ⌘ to record",
    "recording":    "Recording…",
}


def _find_local_models() -> list:
    """Return mlx-community whisper models already cached on disk."""
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    models = []
    if cache.exists():
        for d in sorted(cache.iterdir()):
            name = d.name
            if name.startswith("models--mlx-community--whisper-"):
                repo = name[len("models--"):].replace("--", "/")
                models.append(repo)
    return models if models else [MODEL]


def _normalize_command_text(text: str) -> str:
    text = text.lower()
    text = (
        text.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _command_variants(text: str) -> list[str]:
    variants = [text]
    parts = text.split()
    if parts:
        first = parts[0]
        if first.endswith("es") and len(first) > 3:
            variants.append(" ".join([first[:-2], *parts[1:]]).strip())
        if first.endswith("s") and len(first) > 2:
            variants.append(" ".join([first[:-1], *parts[1:]]).strip())
    return [variant for i, variant in enumerate(variants) if variant and variant not in variants[:i]]


def _strip_payload_prefix(text: str) -> str:
    for prefix in ("and write ", "write ", "and type ", "type ", "and "):
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text.strip()


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        curr = [i]
        for j, right in enumerate(b, start=1):
            cost = 0 if left == right else 1
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = curr
    return prev[-1]


def _is_fuzzy_match(left: str, right: str) -> bool:
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return False
    if left == right:
        return True
    max_len = max(len(left), len(right))
    max_edits = max(1, min(3, math.ceil(max_len * 0.3)))
    if left.startswith(right) or right.startswith(left):
        return abs(len(left) - len(right)) <= max_edits
    return _edit_distance(left, right) <= max_edits


def _extract_fuzzy_prefix(candidate: str, trigger: str):
    candidate_words = candidate.split()
    trigger_words = trigger.split()
    if not candidate_words or not trigger_words:
        return None

    sizes = []
    for size in (len(trigger_words) - 1, len(trigger_words), len(trigger_words) + 1):
        if 1 <= size <= len(candidate_words) and size not in sizes:
            sizes.append(size)

    for size in sizes:
        prefix = " ".join(candidate_words[:size])
        if _is_fuzzy_match(prefix, trigger):
            return " ".join(candidate_words[size:]).strip()

    if trigger in candidate:
        return candidate.split(trigger, 1)[1].strip()
    return None


class MyWhisper:
    def __init__(self):
        self.state         = "loading"
        self._model        = MODEL
        self._local_models = _find_local_models()
        self._actions      = []
        self._actions_mtime = None
        self._wake_words   = []
        self._task         = TASK            # can be toggled live
        self._chunks       = []
        self._recording = False
        self._rms       = 0.0
        self._levels    = [0.0] * BAR_N   # smoothed display levels
        self._phase        = 0.0              # animation phase for loading/transcribing
        self._last_time    = None            # seconds taken for last transcription
        self._cur_scale    = 1.0             # current animated window scale (1.0 = normal)
        self._shrink_after = 0.0             # timestamp after which to start shrinking
        self._launcher_mode = False          # True while waiting for a launcher keypress
        self._launcher_keys: dict = {}       # char → action, built by _assign_launcher_keys
        self._launcher_win  = None
        self._press_time    = 0.0
        self._kb           = KBController()

        self._build_ui()
        self._reload_actions(force=True)
        self._start_audio()
        self._start_hotkey()
        threading.Thread(target=self._load_model, daemon=True).start()
        self._tick()
        self.root.mainloop()

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.94)
        root.configure(bg=BG)
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.geometry(f"{WIN_W}x{WIN_H}+{sw - WIN_W - 24}+{sh - WIN_H - 24}")
        self.root = root
        self._sw = sw
        self._sh = sh

        cv = tk.Canvas(root, width=WIN_W, height=WIN_H, bg=BG, highlightthickness=0)
        cv.pack()
        self.cv = cv

        PAD = 14
        # Status dot
        self._dot = cv.create_oval(PAD, 11, PAD + 9, 20, fill="#475569", outline="")
        # Status label
        self._lbl = cv.create_text(PAD + 16, 15, text=LABELS_BASE["loading"],
                                   fill="#475569", font=("Helvetica Neue", 11),
                                   anchor="w")
        # Model name — right-aligned, dim; clickable when multiple local models exist
        short_model = self._model.split("/")[-1]
        model_col = "#475569" if len(self._local_models) > 1 else "#334155"
        self._model_lbl = cv.create_text(WIN_W - 68, 15, text=short_model, fill=model_col,
                                         font=("Helvetica Neue", 10), anchor="e", tags="model")
        if len(self._local_models) > 1:
            cv.tag_bind("model", "<Button-1>", lambda _: self._cycle_model())
            cv.tag_bind("model", "<Enter>",    lambda _: cv.itemconfig("model", fill="#94a3b8"))
            cv.tag_bind("model", "<Leave>",    lambda _: cv.itemconfig("model", fill="#475569"))

        # Translate toggle
        self._translate_lbl = cv.create_text(WIN_W - 40, 15, text="",
                                             font=("Helvetica Neue", 10), anchor="e", tags="translate")
        cv.tag_bind("translate", "<Button-1>", lambda _: self._toggle_translate())
        cv.tag_bind("translate", "<Enter>",    lambda _: cv.itemconfig("translate", fill="#94a3b8"))
        cv.tag_bind("translate", "<Leave>",    lambda _: self._update_translate_lbl())

        # Gear icon — opens settings
        cv.create_text(WIN_W - 24, 15, text="⚙", fill="#334155",
                       font=("Helvetica Neue", 12), anchor="e", tags="gear")
        cv.tag_bind("gear", "<Button-1>", lambda _: self._open_settings())
        cv.tag_bind("gear", "<Enter>",    lambda _: cv.itemconfig("gear", fill="#94a3b8"))
        cv.tag_bind("gear", "<Leave>",    lambda _: cv.itemconfig("gear", fill="#334155"))

        # Close ×
        cv.create_text(WIN_W - 10, 14, text="×", fill="#334155",
                       font=("Helvetica Neue", 14), anchor="e", tags="close")
        cv.tag_bind("close", "<Button-1>",  lambda _: self._quit())
        cv.tag_bind("close", "<Enter>",     lambda _: cv.itemconfig("close", fill="#94a3b8"))
        cv.tag_bind("close", "<Leave>",     lambda _: cv.itemconfig("close", fill="#334155"))

        # Level bars
        BAR_TOP = 30
        BAR_BOT = WIN_H - 8
        bar_w   = (WIN_W - 2 * PAD) / BAR_N
        self._bars = []
        for i in range(BAR_N):
            x0 = PAD + i * bar_w + 0.8
            x1 = x0 + bar_w - 1.6
            b  = cv.create_rectangle(x0, BAR_BOT - 2, x1, BAR_BOT,
                                     fill="#1e293b", outline="", width=0)
            self._bars.append(b)
        self._BAR_TOP = BAR_TOP
        self._BAR_BOT = BAR_BOT
        self._BAR_W   = bar_w
        self._PAD     = PAD

        # Drag
        cv.bind("<ButtonPress-1>", lambda e: setattr(self, "_drag", (e.x, e.y)))
        cv.bind("<B1-Motion>",     self._on_drag)

        # Right-click opens settings
        cv.bind("<Button-2>", lambda e: self._open_settings())
        cv.bind("<Button-3>", lambda e: self._open_settings())

    def _on_drag(self, e):
        dx, dy = self._drag
        self.root.geometry(f"+{self.root.winfo_x() + e.x - dx}+{self.root.winfo_y() + e.y - dy}")

    def _toggle_translate(self):
        self._task = "translate" if self._task == "transcribe" else "transcribe"
        self._update_translate_lbl()

    def _update_translate_lbl(self):
        if self._task == "translate":
            self.cv.itemconfig(self._translate_lbl, text="→EN", fill="#f59e0b")
        else:
            self.cv.itemconfig(self._translate_lbl, text="→EN", fill="#334155")

    def _cycle_model(self):
        if self.state != "idle" or len(self._local_models) <= 1:
            return
        idx = self._local_models.index(self._model) if self._model in self._local_models else 0
        self._model = self._local_models[(idx + 1) % len(self._local_models)]
        self._last_time = None
        self.cv.itemconfig(self._model_lbl, text=self._model.split("/")[-1])
        self.state = "loading"
        threading.Thread(target=self._load_model, daemon=True).start()

    def _open_settings(self):
        # Singleton — raise existing window if already open
        if hasattr(self, "_settings_win") and self._settings_win and self._settings_win.winfo_exists():
            self._settings_win.lift()
            self._settings_win.focus_force()
            return

        win = tk.Toplevel(self.root)
        win.title("mywhisper Settings")
        win.configure(bg="#0f172a")
        win.resizable(False, False)
        win.attributes("-topmost", True)
        self._settings_win = win

        # ── ttk dark style ────────────────────────────────────────────────────
        style = ttk.Style(win)
        style.theme_use("clam")
        for name, cfg in {
            "Dark.TFrame":       {"background": "#0f172a"},
            "Head.TLabel":       {"background": "#0f172a", "foreground": "#e2e8f0",
                                  "font": ("Helvetica Neue", 13, "bold")},
            "Dark.TLabel":       {"background": "#0f172a", "foreground": "#94a3b8",
                                  "font": ("Helvetica Neue", 12)},
            "Key.TLabel":        {"background": "#0f172a", "foreground": "#cbd5e1",
                                  "font": ("Helvetica Neue", 11, "bold")},
            "Desc.TLabel":       {"background": "#0f172a", "foreground": "#64748b",
                                  "font": ("Helvetica Neue", 11)},
            "Dark.TCheckbutton": {"background": "#0f172a", "foreground": "#94a3b8",
                                  "font": ("Helvetica Neue", 12)},
        }.items():
            style.configure(name, **cfg)
        style.configure("Dark.TButton", background="#1e293b", foreground="#94a3b8",
                        font=("Helvetica Neue", 11), borderwidth=0, relief="flat")
        style.map("Dark.TButton", background=[("active", "#334155")])
        style.configure("Dark.TCombobox", fieldbackground="#1e293b", background="#1e293b",
                        foreground="#e2e8f0", selectbackground="#1e293b", selectforeground="#e2e8f0")
        style.map("Dark.TCombobox", fieldbackground=[("readonly", "#1e293b")],
                  foreground=[("readonly", "#e2e8f0")])
        style.configure("TSeparator", background="#1e293b")

        row = 0

        # ── Settings ─────────────────────────────────────────────────────────
        ttk.Label(win, text="Settings", style="Head.TLabel").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=16, pady=(14, 6))
        row += 1

        ttk.Label(win, text="Model", style="Dark.TLabel").grid(
            row=row, column=0, sticky="w", padx=(16, 8), pady=4)
        model_var = tk.StringVar(value=self._model)
        combo = ttk.Combobox(win, textvariable=model_var, values=self._local_models,
                             state="readonly", width=36, style="Dark.TCombobox")
        combo.grid(row=row, column=1, sticky="w", padx=(0, 16), pady=4)
        row += 1

        def _on_model_change(event=None):
            new_model = model_var.get()
            if new_model != self._model and self.state == "idle":
                self._model = new_model
                self.cv.itemconfig(self._model_lbl, text=self._model.split("/")[-1])
                self._last_time = None
                self.state = "loading"
                threading.Thread(target=self._load_model, daemon=True).start()
        combo.bind("<<ComboboxSelected>>", _on_model_change)

        translate_var = tk.BooleanVar(value=self._task == "translate")

        def _on_translate_toggle():
            self._task = "translate" if translate_var.get() else "transcribe"
            self._update_translate_lbl()

        ttk.Checkbutton(win, text="Translate to English", variable=translate_var,
                        command=_on_translate_toggle, style="Dark.TCheckbutton").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=16, pady=(2, 8))
        row += 1

        # ── Separator ────────────────────────────────────────────────────────
        ttk.Separator(win, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", padx=16, pady=2)
        row += 1

        # ── Help / Shortcuts ─────────────────────────────────────────────────
        ttk.Label(win, text="Keyboard Shortcuts", style="Head.TLabel").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=16, pady=(10, 4))
        row += 1

        shortcuts = [
            ("Hold right ⌘",           "Record audio"),
            ("Tap right ⌘",            "Open action launcher"),
            ("Click ⚙  or right-click", "Open settings"),
            ("Click →EN",              "Toggle translate to English"),
            ("Click model name",        "Cycle to next cached model"),
            ("Click ×",                "Quit"),
        ]
        for key, desc in shortcuts:
            ttk.Label(win, text=key, style="Key.TLabel").grid(
                row=row, column=0, sticky="w", padx=(16, 12), pady=2)
            ttk.Label(win, text=desc, style="Desc.TLabel").grid(
                row=row, column=1, sticky="w", padx=(0, 16), pady=2)
            row += 1

        # ── Separator ────────────────────────────────────────────────────────
        ttk.Separator(win, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", padx=16, pady=8)
        row += 1

        # ── Bottom buttons ───────────────────────────────────────────────────
        ttk.Button(win, text="GitHub ↗", style="Dark.TButton",
                   command=lambda: webbrowser.open(GITHUB_URL)).grid(
            row=row, column=0, sticky="w", padx=16, pady=(0, 14))
        ttk.Button(win, text="Close", style="Dark.TButton",
                   command=win.destroy).grid(
            row=row, column=1, sticky="e", padx=16, pady=(0, 14))

        # Position above the overlay (overlay is anchored bottom-right)
        win.update_idletasks()
        ox, oy = self.root.winfo_x(), self.root.winfo_y()
        win_h = win.winfo_reqheight()
        win.geometry(f"+{ox}+{max(0, oy - win_h - 4)}")
        win.lift()
        win.focus_force()

    # ── Audio ──────────────────────────────────────────────────────────────────

    def _start_audio(self):
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=512, callback=self._audio_cb,
        )
        self._stream.start()

    def _audio_cb(self, indata, frames, t, status):
        mono = indata[:, 0]
        self._rms = float(np.sqrt(np.mean(mono ** 2)))
        if self._recording:
            self._chunks.append(mono.copy())

    # ── Model ──────────────────────────────────────────────────────────────────

    def _load_model(self):
        # Warm-up: compiles the graph so first real transcription is fast
        mlx_whisper.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32),
                               path_or_hf_repo=self._model, task=self._task, verbose=False)
        self._update_translate_lbl()
        self.state = "idle"

    def _reload_actions(self, force: bool = False):
        if not ACTIONS_FILE.exists():
            self._actions = []
            self._actions_mtime = None
            return

        mtime = ACTIONS_FILE.stat().st_mtime
        if not force and self._actions_mtime == mtime:
            return

        try:
            data = json.loads(ACTIONS_FILE.read_text())
            actions = data.get("actions", data) if isinstance(data, dict) else data
            wake_words = data.get("wake_words", []) if isinstance(data, dict) else []
            loaded = []
            for action in actions:
                if not isinstance(action, dict):
                    continue
                trigger = _normalize_command_text(str(action.get("trigger", "")).strip())
                action_type = str(action.get("type", "")).strip().lower()
                target = str(action.get("target", "")).strip()
                if trigger and action_type in {"open", "shell"} and target:
                    loaded.append(
                        {
                            "label": str(action.get("label", trigger)).strip(),
                            "trigger": trigger,
                            "type": action_type,
                            "target": target,
                            "paste_result": bool(action.get("paste_result", False)),
                            "new_chat": bool(action.get("new_chat", False)),
                            "key": str(action.get("key", "")).strip().lower()[:1],
                        }
                    )
            self._actions = loaded
            self._wake_words = [
                _normalize_command_text(str(wake_word))
                for wake_word in wake_words
                if _normalize_command_text(str(wake_word))
            ]
            self._actions = loaded
            self._assign_launcher_keys()
            self._actions_mtime = mtime
        except Exception as e:
            print(f"[mywhisper] actions.json error: {e}")

    def _assign_launcher_keys(self):
        """Auto-assign a single letter to each action for the tap launcher."""
        used: set = set()
        for action in self._actions:
            explicit = action.get("key", "").strip().lower()
            if explicit and explicit.isalpha() and explicit not in used:
                action["_key"] = explicit
                used.add(explicit)
                continue
            # Auto-assign: first unused letter in the label
            assigned = None
            for ch in action.get("label", "").lower():
                if ch.isalpha() and ch not in used:
                    assigned = ch
                    break
            action["_key"] = assigned
            if assigned:
                used.add(assigned)
        self._launcher_keys = {a["_key"]: a for a in self._actions if a.get("_key")}

    def _match_action(self, text: str):
        haystack = _normalize_command_text(text)
        if self._wake_words:
            candidates = []
            words = haystack.split()
            if words:
                spoken_wake = words[0]
                for wake_word in self._wake_words:
                    if _is_fuzzy_match(spoken_wake, wake_word):
                        candidate = " ".join(words[1:]).strip()
                        if candidate:
                            candidates.extend(_command_variants(candidate))
        else:
            candidates = _command_variants(haystack)
        for action in self._actions:
            for candidate in candidates:
                payload = _extract_fuzzy_prefix(candidate, action["trigger"])
                if payload is not None:
                    payload = _strip_payload_prefix(payload)
                    return action, payload
        return None, ""

    def _run_action(self, action, text: str, payload: str = ""):
        try:
            app_name = None
            if action["type"] == "open":
                target = action["target"]
                local_target = None
                if not any(target.startswith(prefix) for prefix in ("http://", "https://", "tel:", "mailto:", "obsidian:")):
                    local_target = Path(target).expanduser()
                    app_name = local_target.stem if local_target.suffix == ".app" else None
                    target = str(local_target)
                subprocess.Popen(["open", target])
                if local_target is not None and local_target.exists() and local_target.suffix != ".app":
                    subprocess.Popen(["osascript", "-e", 'tell application "Finder" to activate'])
            elif action["type"] == "shell":
                subprocess.Popen(action["target"], shell=True)
            if action.get("new_chat") and app_name:
                time.sleep(0.5)
                subprocess.Popen(["osascript", "-e", f'tell application "{app_name}" to activate'])
                time.sleep(0.35)
                self._kb.press(Key.cmd)
                self._kb.press("n")
                self._kb.release("n")
                self._kb.release(Key.cmd)
            if action["paste_result"]:
                if payload:
                    time.sleep(0.8 if action.get("new_chat") else 0.35)
                    self._paste(payload)
            print(f"[mywhisper] action: {action['label']}")
        except Exception as e:
            print(f"[mywhisper] action error: {e}")

    # ── Hotkey ─────────────────────────────────────────────────────────────────

    def _start_hotkey(self):
        def on_press(key):
            # Launcher mode: route all keys through the launcher (pynput-side, no focus needed)
            if self._launcher_mode:
                self.root.after(0, lambda k=key: self._launcher_key(k))
                return
            if key == HOTKEY and self.state == "idle":
                self._press_time = time.time()
                self._chunks = []
                self._recording = True
                self.state = "recording"

        def on_release(key):
            if key == HOTKEY and self._recording:
                self._recording = False
                duration = time.time() - self._press_time
                if duration < 0.35 and self._launcher_keys:
                    # Short tap → open launcher instead of transcribing
                    self._launcher_mode = True
                    self.root.after(0, self._open_launcher)
                else:
                    # Hold → transcribe as normal
                    self.state = "transcribing"
                    threading.Thread(target=self._transcribe, daemon=True).start()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    # ── Launcher ───────────────────────────────────────────────────────────────

    def _open_launcher(self):
        if self._launcher_win and self._launcher_win.winfo_exists():
            return
        win = tk.Toplevel(self.root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.attributes("-alpha", 0.96)
        win.configure(bg=BG)
        self._launcher_win = win

        ROW_H = 28
        PAD   = 12
        items = [(a["_key"], a["label"]) for a in self._actions if a.get("_key")]
        # Width based on longest label
        max_label = max((len(lbl) for _, lbl in items), default=10)
        win_w = max(200, PAD + 34 + max_label * 7 + PAD)
        win_h = PAD + len(items) * ROW_H + 20

        cv = tk.Canvas(win, width=win_w, height=win_h, bg=BG, highlightthickness=0)
        cv.pack()

        for i, (key_char, label) in enumerate(items):
            y = PAD + i * ROW_H + ROW_H // 2
            cv.create_text(PAD, y, text=f"[{key_char}]",
                           fill="#f59e0b", font=("Helvetica Neue", 11, "bold"), anchor="w")
            cv.create_text(PAD + 32, y, text=label,
                           fill="#e2e8f0", font=("Helvetica Neue", 11), anchor="w")

        cv.create_text(win_w // 2, win_h - 5, text="esc · cancel",
                       fill="#334155", font=("Helvetica Neue", 9), anchor="s")

        # Center on screen
        win.update_idletasks()
        ox = (self._sw - win_w) // 2
        oy = (self._sh - win_h) // 2
        win.geometry(f"{win_w}x{win_h}+{ox}+{oy}")

        # Auto-close after 6 s if nothing pressed
        self._launcher_timeout = self.root.after(6000, self._close_launcher)

    def _launcher_key(self, key):
        if not self._launcher_mode:
            return
        if key == keyboard.Key.esc:
            self._close_launcher()
            return
        char = getattr(key, "char", None)
        if char:
            action = self._launcher_keys.get(char.lower())
            if action:
                self._close_launcher()
                threading.Thread(target=self._run_action, args=(action, "", ""), daemon=True).start()
                return
        # Unmapped key → dismiss
        self._close_launcher()

    def _close_launcher(self):
        self._launcher_mode = False
        if hasattr(self, "_launcher_timeout"):
            try:
                self.root.after_cancel(self._launcher_timeout)
            except Exception:
                pass
        if self._launcher_win:
            try:
                self._launcher_win.destroy()
            except Exception:
                pass
            self._launcher_win = None
        # Discard the tiny audio captured during the tap
        self._chunks = []
        self.state = "idle"

    # ── Transcribe & paste ─────────────────────────────────────────────────────

    def _transcribe(self):
        if not self._chunks:
            self.state = "idle"
            return
        audio = np.concatenate(self._chunks).astype(np.float32)
        try:
            self._reload_actions()
            t0 = time.time()
            result = mlx_whisper.transcribe(audio, path_or_hf_repo=self._model, task=self._task, verbose=False)
            self._last_time = time.time() - t0
            text = result.get("text", "").strip()
            if text:
                action, payload = self._match_action(text)
                if action is not None:
                    self._run_action(action, text, payload)
                else:
                    self._paste(text)
        except Exception as e:
            print(f"[mywhisper] transcription error: {e}")
        self.state = "idle"

    def _paste(self, text: str):
        pyperclip.copy(text)
        time.sleep(0.05)
        self._kb.press(Key.cmd)
        self._kb.press("v")
        self._kb.release("v")
        self._kb.release(Key.cmd)

    # ── Render loop ────────────────────────────────────────────────────────────

    def _tick(self):
        self._phase = (self._phase + 4) % 360

        # ── Window scale animation ─────────────────────────────────────────
        now = time.time()
        if self.state in ("recording", "transcribing"):
            self._shrink_after = now + 1.0   # keep large for 1 s after state ends
            target_scale = 2.0
        elif now < self._shrink_after:
            target_scale = 2.0               # hold period after transcription
        else:
            target_scale = 1.0
        self._cur_scale += (target_scale - self._cur_scale) * 0.15
        s = self._cur_scale

        # ── Resize window, anchored to bottom-right corner ─────────────────
        cw = max(1, int(WIN_W * s))
        ch = max(1, int(WIN_H * s))
        self.root.geometry(f"{cw}x{ch}+{self._sw - cw - 24}+{self._sh - ch - 24}")
        self.cv.configure(width=cw, height=ch)

        # ── Scaled layout constants ────────────────────────────────────────
        PAD     = 14 * s
        BAR_TOP = 30 * s
        BAR_BOT = (WIN_H - 8) * s
        bar_w   = (WIN_W - 28) * s / BAR_N   # 28 = 2 * PAD_BASE
        bar_max = BAR_BOT - BAR_TOP

        # ── Reposition header elements ─────────────────────────────────────
        dot_col, txt_col, bar_base = STATES[self.state]
        self.cv.coords(self._dot, PAD, 11 * s, PAD + 9 * s, 20 * s)
        self.cv.coords(self._lbl, PAD + 16 * s, 15 * s)
        self.cv.coords(self._model_lbl, (WIN_W - 68) * s, 15 * s)
        self.cv.coords(self._translate_lbl, (WIN_W - 40) * s, 15 * s)
        for tag, rx, ry in (("gear", WIN_W - 24, 15), ("close", WIN_W - 10, 14)):
            items = self.cv.find_withtag(tag)
            if items:
                self.cv.coords(items[0], rx * s, ry * s)

        # ── Bar drawing ────────────────────────────────────────────────────
        if self.state in ("loading", "transcribing"):
            # Animated sine wave
            for i, bar in enumerate(self._bars):
                wave = 0.5 + 0.5 * math.sin(math.radians(self._phase + i * (360 / BAR_N)))
                h    = max(2, int(wave * bar_max * 0.55))
                x0   = PAD + i * bar_w + 0.8
                x1   = x0 + bar_w - 1.6
                self.cv.coords(bar, x0, BAR_BOT - h, x1, BAR_BOT)
                self.cv.itemconfig(bar, fill=bar_base)
        else:
            # Live level meter — smooth towards current RMS
            rms = self._rms
            self._levels = [v * 0.75 + (rms if i == BAR_N - 1 else self._levels[min(i + 1, BAR_N - 1)]) * 0.25
                            for i, v in enumerate(self._levels)]
            # Scroll: push new value in at right
            self._levels = self._levels[1:] + [self._levels[-1] * 0.8 + rms * 0.2]

            for i, (bar, level) in enumerate(zip(self._bars, self._levels)):
                h  = max(2, min(bar_max, int(level * bar_max * 18)))
                x0 = PAD + i * bar_w + 0.8
                x1 = x0 + bar_w - 1.6
                self.cv.coords(bar, x0, BAR_BOT - h, x1, BAR_BOT)

                if self.state == "recording":
                    # Green → red gradient based on level
                    t   = min(1.0, level * 22)
                    r   = int(34  + (239 - 34)  * t)
                    g   = int(197 - (197 - 68)  * t)
                    b   = int(94  - 94           * t)
                    col = f"#{r:02x}{g:02x}{b:02x}"
                else:
                    # Idle: dim blue, brighter with level
                    t   = min(1.0, level * 22)
                    r   = int(30  + 44  * t)
                    g   = int(58  + 96  * t)
                    b   = int(95  + 160 * t)
                    col = f"#{r:02x}{g:02x}{b:02x}"
                self.cv.itemconfig(bar, fill=col)

        self.cv.itemconfig(self._dot, fill=dot_col)
        if self.state == "transcribing":
            lbl_text = "Translating…" if self._task == "translate" else "Transcribing…"
        else:
            lbl_text = LABELS_BASE.get(self.state, "")
        self.cv.itemconfig(self._lbl, text=lbl_text, fill=txt_col)
        short_model = self._model.split("/")[-1]
        if self._last_time is not None:
            self.cv.itemconfig(self._model_lbl, text=f"{self._last_time:.1f}s  {short_model}")
        self.root.after(1000 // FPS, self._tick)

    # ── Quit ───────────────────────────────────────────────────────────────────

    def _quit(self):
        self._stream.stop()
        self._listener.stop()
        self.root.destroy()


if __name__ == "__main__":
    MyWhisper()
