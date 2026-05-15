#!/usr/bin/env python3
"""
mychinesewhisper — hold right-⌘ to record, release to transcribe and translate to Chinese.

Two-step pipeline (fully offline, no API):
  1. mlx_whisper  — transcribes speech in original language (fast, Apple Silicon)
  2. Helsinki-NLP/opus-mt-de-zh — translates German text → Simplified Chinese (~50ms)

Usage:
    python mychinesewhisper.py
    python mychinesewhisper.py --model mlx-community/whisper-large-v3-turbo
"""

import argparse
import math
import threading
import time
import tkinter as tk

import numpy as np
import pyperclip
import sounddevice as sd
import mlx_whisper
from pynput import keyboard
from pynput.keyboard import Controller as KBController, Key
from transformers import MarianMTModel, MarianTokenizer

# ── CLI ────────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(
    description="mychinesewhisper — record speech, translate to Chinese, paste at cursor",
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
        "  mlx-community/whisper-large-v3-turbo     (best quality, no -mlx suffix)\n"
    ),
)
_args = _parser.parse_args()

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL       = _args.model
MT_MODEL    = "Helsinki-NLP/opus-mt-de-zh"   # German → Simplified Chinese
HOTKEY      = Key.cmd_r
SAMPLE_RATE = 16000
WIN_W       = 360
WIN_H       = 72
BAR_N       = 48
FPS         = 30

# ── Palette ────────────────────────────────────────────────────────────────────
BG   = "#0f172a"
STATES = {
    "loading":      ("#475569", "#475569", "#1e293b"),
    "idle":         ("#22c55e", "#64748b", "#1e3a5f"),
    "recording":    ("#ef4444", "#ef4444", "#7f1d1d"),
    "transcribing": ("#f59e0b", "#f59e0b", "#78350f"),
}
LABELS_BASE = {
    "loading":   "Loading models…",
    "idle":      "Ready  ·  hold right ⌘ to record",
    "recording": "Recording…",
}


class MyChineseWhisper:
    def __init__(self):
        self.state      = "loading"
        self._chunks    = []
        self._recording = False
        self._rms       = 0.0
        self._levels    = [0.0] * BAR_N
        self._phase     = 0.0
        self._last_time = None
        self._kb        = KBController()
        self._mt_tok    = None   # MarianTokenizer, loaded in background
        self._mt_model  = None   # MarianMTModel, loaded in background

        self._build_ui()
        self._start_audio()
        self._start_hotkey()
        threading.Thread(target=self._load_models, daemon=True).start()
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
        root.geometry(f"{WIN_W}x{WIN_H}+{sw - WIN_W - 24}+24")
        self.root = root

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
        # Model name — right-aligned, dim
        short_model = MODEL.split("/")[-1]
        self._model_lbl = cv.create_text(WIN_W - 52, 15, text=short_model, fill="#334155",
                                         font=("Helvetica Neue", 10), anchor="e")

        # Static →ZH indicator (always active — this app always translates to Chinese)
        self._zh_lbl = cv.create_text(WIN_W - 26, 15, text="→ZH",
                                      fill="#334155",   # dim until model loaded
                                      font=("Helvetica Neue", 10), anchor="e")

        # Close ×
        cv.create_text(WIN_W - 10, 14, text="×", fill="#334155",
                       font=("Helvetica Neue", 14), anchor="e", tags="close")
        cv.tag_bind("close", "<Button-1>", lambda _: self._quit())
        cv.tag_bind("close", "<Enter>",    lambda _: cv.itemconfig("close", fill="#94a3b8"))
        cv.tag_bind("close", "<Leave>",    lambda _: cv.itemconfig("close", fill="#334155"))

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

    def _on_drag(self, e):
        dx, dy = self._drag
        self.root.geometry(f"+{self.root.winfo_x() + e.x - dx}+{self.root.winfo_y() + e.y - dy}")

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

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_models(self):
        # 1. Warm-up Whisper (compiles MLX graph)
        mlx_whisper.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32),
                               path_or_hf_repo=MODEL, task="transcribe", verbose=False)

        # 2. Load MarianMT translation model (downloads ~300MB on first run, then cached)
        self._mt_tok   = MarianTokenizer.from_pretrained(MT_MODEL)
        self._mt_model = MarianMTModel.from_pretrained(MT_MODEL)

        # Light up →ZH indicator
        self.root.after(0, lambda: self.cv.itemconfig(self._zh_lbl, fill="#22d3ee"))
        self.state = "idle"

    # ── Translation ────────────────────────────────────────────────────────────

    def _to_chinese(self, text: str) -> str:
        """Translate text to Simplified Chinese via Helsinki-NLP MarianMT."""
        tokens = self._mt_tok([text], return_tensors="pt", padding=True, truncation=True)
        out    = self._mt_model.generate(**tokens)
        return self._mt_tok.decode(out[0], skip_special_tokens=True)

    # ── Hotkey ─────────────────────────────────────────────────────────────────

    def _start_hotkey(self): 
        def on_press(key):
            if key == HOTKEY and self.state == "idle":
                self._chunks = []
                self._recording = True
                self.state = "recording"

        def on_release(key):
            if key == HOTKEY and self._recording:
                self._recording = False
                self.state = "transcribing"
                threading.Thread(target=self._transcribe, daemon=True).start()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    # ── Transcribe → translate → paste ────────────────────────────────────────

    def _transcribe(self):
        if not self._chunks:
            self.state = "idle"
            return
        audio = np.concatenate(self._chunks).astype(np.float32)
        try:
            t0     = time.time()
            result = mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL,
                                            task="transcribe", verbose=False)
            german = result.get("text", "").strip()
            if german:
                chinese = self._to_chinese(german)
                self._last_time = time.time() - t0
                self._paste(chinese)
        except Exception as e:
            print(f"[mychinesewhisper] error: {e}")
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
        dot_col, txt_col, bar_base = STATES[self.state]
        PAD     = self._PAD
        BAR_TOP = self._BAR_TOP
        BAR_BOT = self._BAR_BOT
        bar_w   = self._BAR_W
        bar_max = BAR_BOT - BAR_TOP

        if self.state in ("loading", "transcribing"):
            for i, bar in enumerate(self._bars):
                wave = 0.5 + 0.5 * math.sin(math.radians(self._phase + i * (360 / BAR_N)))
                h    = max(2, int(wave * bar_max * 0.55))
                x0   = PAD + i * bar_w + 0.8
                x1   = x0 + bar_w - 1.6
                self.cv.coords(bar, x0, BAR_BOT - h, x1, BAR_BOT)
                self.cv.itemconfig(bar, fill=bar_base)
        else:
            rms = self._rms
            self._levels = self._levels[1:] + [self._levels[-1] * 0.8 + rms * 0.2]

            for i, (bar, level) in enumerate(zip(self._bars, self._levels)):
                h  = max(2, min(bar_max, int(level * bar_max * 18)))
                x0 = PAD + i * bar_w + 0.8
                x1 = x0 + bar_w - 1.6
                self.cv.coords(bar, x0, BAR_BOT - h, x1, BAR_BOT)

                if self.state == "recording":
                    t   = min(1.0, level * 22)
                    r   = int(34  + (239 - 34)  * t)
                    g   = int(197 - (197 - 68)  * t)
                    b   = int(94  - 94           * t)
                    col = f"#{r:02x}{g:02x}{b:02x}"
                else:
                    t   = min(1.0, level * 22)
                    r   = int(30  + 44  * t)
                    g   = int(58  + 96  * t)
                    b   = int(95  + 160 * t)
                    col = f"#{r:02x}{g:02x}{b:02x}"
                self.cv.itemconfig(bar, fill=col)

        self.cv.itemconfig(self._dot, fill=dot_col)
        if self.state == "transcribing":
            lbl_text = "Translating → 中文…"
        else:
            lbl_text = LABELS_BASE.get(self.state, "")
        self.cv.itemconfig(self._lbl, text=lbl_text, fill=txt_col)
        short_model = MODEL.split("/")[-1]
        if self._last_time is not None:
            self.cv.itemconfig(self._model_lbl, text=f"{self._last_time:.1f}s  {short_model}")
        self.root.after(1000 // FPS, self._tick)

    # ── Quit ───────────────────────────────────────────────────────────────────

    def _quit(self):
        self._stream.stop()
        self._listener.stop()
        self.root.destroy()


if __name__ == "__main__":
    MyChineseWhisper()
