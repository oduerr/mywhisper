#!/usr/bin/env python3
"""
openai_translate.py — hold right-⌥ to record German speech, release to translate,
hear the translation spoken aloud, and get the text pasted at the cursor.

Usage:
    python openai_translate.py
    python openai_translate.py --lang en   # German → English
    python openai_translate.py --lang fr   # German → French
    python openai_translate.py --lang zh   # German → Chinese (default)
    python openai_translate.py --help

Requires: OPENAI_API_KEY environment variable
macOS:    grant Accessibility permission to your terminal (same as mywhisper.py)
"""

import argparse
import asyncio
import base64
import json
import math
import os
import threading
import time
import tkinter as tk

import numpy as np
import pyperclip
import sounddevice as sd
import websockets
from pynput import keyboard
from pynput.keyboard import Controller as KBController, Key

# ── CLI ────────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(
    description="openai_translate — hold right-⌥, speak German, release to translate and paste",
    formatter_class=argparse.RawTextHelpFormatter,
)
_parser.add_argument(
    "--lang", default="zh", choices=["en", "fr", "zh"],
    help=(
        "Target language (default: zh)\n"
        "  en  German → English\n"
        "  fr  German → French\n"
        "  zh  German → Chinese"
    ),
)
_args = _parser.parse_args()

# ── Config ─────────────────────────────────────────────────────────────────────
TARGET_LANG = _args.lang
SAMPLE_RATE = 24000        # PCM16 24 kHz required by OpenAI Realtime API
HOTKEY      = Key.alt_r
WIN_W       = 380
BAR_N       = 48
FPS         = 30
TEXT_H      = 44           # reserved height below bars for translated text

LANG_LABEL  = {"en": "DE → English", "fr": "DE → Français", "zh": "DE → 中文"}
LANG_NAMES  = {"en": "English", "fr": "French", "zh": "Chinese"}

# ── Palette ────────────────────────────────────────────────────────────────────
BG = "#0f172a"
STATES = {
    #               dot        status     bar base
    "idle":        ("#22c55e", "#64748b", "#1e3a5f"),
    "recording":   ("#ef4444", "#ef4444", "#7f1d1d"),
    "translating": ("#f59e0b", "#f59e0b", "#78350f"),
    "playing":     ("#a855f7", "#a855f7", "#3b0764"),
}
LABELS = {
    "idle":        "Ready  ·  hold right ⌥ to record",
    "recording":   "Recording…",
    "translating": "Translating…",
    "playing":     "Playing translation…",
}

# ── Derived layout ─────────────────────────────────────────────────────────────
_BAR_TOP = 28
_BAR_BOT = 70
WIN_H    = _BAR_BOT + TEXT_H + 10


class OpenAITranslate:
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

        self.state      = "idle"
        self._chunks    = []
        self._recording = False
        self._rms       = 0.0
        self._levels    = [0.0] * BAR_N
        self._phase     = 0.0
        self._text      = ""          # last translated text
        self._kb        = KBController()
        self._api_key   = api_key

        self._build_ui()
        self._start_audio()
        self._start_hotkey()
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
        # Status dot + label
        self._dot = cv.create_oval(PAD, 11, PAD + 9, 20, fill=STATES["idle"][0], outline="")
        self._lbl = cv.create_text(PAD + 16, 15, text=LABELS["idle"],
                                   fill=STATES["idle"][1], font=("Helvetica Neue", 11), anchor="w")

        # Language label — right-aligned, dim
        cv.create_text(WIN_W - 26, 15, text=LANG_LABEL[TARGET_LANG],
                       fill="#334155", font=("Helvetica Neue", 10), anchor="e")

        # Close ×
        cv.create_text(WIN_W - 10, 14, text="×", fill="#334155",
                       font=("Helvetica Neue", 14), anchor="e", tags="close")
        cv.tag_bind("close", "<Button-1>", lambda _: self._quit())
        cv.tag_bind("close", "<Enter>",    lambda _: cv.itemconfig("close", fill="#94a3b8"))
        cv.tag_bind("close", "<Leave>",    lambda _: cv.itemconfig("close", fill="#334155"))

        # Level bars
        bar_w = (WIN_W - 2 * PAD) / BAR_N
        self._bars = []
        for i in range(BAR_N):
            x0 = PAD + i * bar_w + 0.8
            x1 = x0 + bar_w - 1.6
            b  = cv.create_rectangle(x0, _BAR_BOT - 2, x1, _BAR_BOT,
                                     fill="#1e293b", outline="", width=0)
            self._bars.append(b)
        self._bar_w = bar_w
        self._PAD   = PAD

        # Divider line between bars and text
        cv.create_line(PAD, _BAR_BOT + 8, WIN_W - PAD, _BAR_BOT + 8,
                       fill="#1e293b", width=1)

        # Translated text display
        self._txt = cv.create_text(
            PAD, _BAR_BOT + 14,
            text="", fill="#475569",
            font=("Helvetica Neue", 11),
            anchor="nw", width=WIN_W - 2 * PAD,
        )

        # Drag
        cv.bind("<ButtonPress-1>", lambda e: setattr(self, "_drag", (e.x, e.y)))
        cv.bind("<B1-Motion>",     self._on_drag)

    def _on_drag(self, e):
        dx, dy = self._drag
        self.root.geometry(f"+{self.root.winfo_x() + e.x - dx}+{self.root.winfo_y() + e.y - dy}")

    # ── Audio capture ──────────────────────────────────────────────────────────

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

    # ── Hotkey ─────────────────────────────────────────────────────────────────

    def _start_hotkey(self):
        def on_press(key):
            if key == HOTKEY and self.state == "idle":
                self._chunks  = []
                self._recording = True
                self.state    = "recording"

        def on_release(key):
            if key == HOTKEY and self._recording:
                self._recording = False
                self.state = "translating"
                threading.Thread(target=self._run_translation, daemon=True).start()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    # ── Translation ────────────────────────────────────────────────────────────

    def _run_translation(self):
        if not self._chunks:
            self.state = "idle"
            return
        asyncio.run(self._translate_async())

    async def _translate_async(self):
        # Convert float32 → PCM16 → base64
        audio_f32 = np.concatenate(self._chunks).astype(np.float32)
        audio_i16 = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)
        audio_b64 = base64.b64encode(audio_i16.tobytes()).decode()

        text_parts  = []
        audio_parts = []

        url = "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            async with websockets.connect(url, additional_headers=headers) as ws:
                await ws.send(json.dumps({
                    "type": "session.update",
                    "session": {"audio": {"output": {"language": TARGET_LANG}}}
                }))
                await ws.send(json.dumps({
                    "type": "session.input_audio_buffer.append",
                    "audio": audio_b64,
                }))
                await ws.send(json.dumps({"type": "session.close"}))

                async for message in ws:
                    event = json.loads(message)
                    t = event.get("type", "")
                    if t == "session.output_transcript.delta":
                        text_parts.append(event.get("delta", ""))
                        self._text = "".join(text_parts).strip()
                    elif t == "session.output_audio.delta":
                        chunk = np.frombuffer(base64.b64decode(event["delta"]), dtype=np.int16)
                        audio_parts.append(chunk)
                    elif t == "session.closed":
                        break

        except Exception as e:
            print(f"[openai_translate] error: {e}")
            self.state = "idle"
            return

        text = "".join(text_parts).strip()
        if text:
            self._text = text
            pyperclip.copy(text)
            self._paste(text)

        if audio_parts:
            self.state = "playing"
            audio_out  = np.concatenate(audio_parts).astype(np.float32) / 32767.0
            sd.play(audio_out, samplerate=SAMPLE_RATE)
            sd.wait()

        self.state = "idle"

    def _paste(self, text: str):
        time.sleep(0.05)
        self._kb.press(Key.cmd)
        self._kb.press("v")
        self._kb.release("v")
        self._kb.release(Key.cmd)

    # ── Render loop ────────────────────────────────────────────────────────────

    def _tick(self):
        self._phase = (self._phase + 4) % 360
        state = self.state
        dot_col, txt_col, bar_base = STATES.get(state, STATES["idle"])
        PAD     = self._PAD
        bar_w   = self._bar_w
        bar_max = _BAR_BOT - _BAR_TOP

        if state in ("translating", "playing"):
            for i, bar in enumerate(self._bars):
                wave = 0.5 + 0.5 * math.sin(math.radians(self._phase + i * (360 / BAR_N)))
                h    = max(2, int(wave * bar_max * 0.55))
                x0   = PAD + i * bar_w + 0.8
                x1   = x0 + bar_w - 1.6
                self.cv.coords(bar, x0, _BAR_BOT - h, x1, _BAR_BOT)
                self.cv.itemconfig(bar, fill=bar_base)
        else:
            rms = self._rms
            self._levels = self._levels[1:] + [self._levels[-1] * 0.8 + rms * 0.2]
            for i, (bar, level) in enumerate(zip(self._bars, self._levels)):
                h  = max(2, min(bar_max, int(level * bar_max * 18)))
                x0 = PAD + i * bar_w + 0.8
                x1 = x0 + bar_w - 1.6
                self.cv.coords(bar, x0, _BAR_BOT - h, x1, _BAR_BOT)
                if state == "recording":
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
        self.cv.itemconfig(self._lbl, text=LABELS.get(state, ""), fill=txt_col)

        # Text: bright white while streaming/done, dim when idle with no text
        if self._text:
            text_col = "#e2e8f0" if state != "idle" else "#64748b"
            self.cv.itemconfig(self._txt, text=self._text, fill=text_col)

        self.root.after(1000 // FPS, self._tick)

    # ── Quit ───────────────────────────────────────────────────────────────────

    def _quit(self):
        self._stream.stop()
        self._listener.stop()
        self.root.destroy()


if __name__ == "__main__":
    OpenAITranslate()
