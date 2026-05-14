#!/usr/bin/env python3
"""
openai_translate.py — hold right-⌥ to speak (any language); translation
streams live to cursor and speakers. On release: full text goes to clipboard.
Click text to replay.

Input language is auto-detected by the API — you can speak any language.
Only the output/target language is configured via --lang.
Note: gpt-realtime-translate does not support voice selection.

Usage:
    python openai_translate.py
    python openai_translate.py --lang en   # → English
    python openai_translate.py --lang fr   # → French
    python openai_translate.py --lang zh   # → Chinese (default)
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
import queue
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
    description="openai_translate — hold right-⌥, speak any language, translation streams live to cursor",
    formatter_class=argparse.RawTextHelpFormatter,
)
_parser.add_argument(
    "--lang", default="zh",
    choices=["en", "fr", "zh", "fi", "es", "it", "pt", "de"],
    help=(
        "Target language (default: zh). Input language is auto-detected.\n"
        "  en  → English\n"
        "  fr  → French\n"
        "  zh  → Chinese\n"
        "  fi  → Finnish\n"
        "  es  → Spanish\n"
        "  it  → Italian\n"
        "  pt  → Portuguese\n"
        "  de  → German"
    ),
)
_args = _parser.parse_args()

# ── Config ─────────────────────────────────────────────────────────────────────
TARGET_LANG = _args.lang
SAMPLE_RATE = 24000
HOTKEY      = Key.alt_r
WIN_W       = 380
BAR_N       = 48
FPS         = 30
TEXT_H      = 44

LANG_LABEL  = {
    "en": "→ English (auto)",
    "fr": "→ Français (auto)",
    "zh": "→ 中文 (auto)",
    "fi": "→ Suomi (auto)",
    "es": "→ Español (auto)",
    "it": "→ Italiano (auto)",
    "pt": "→ Português (auto)",
    "de": "→ Deutsch (auto)",
}

# ── Palette ────────────────────────────────────────────────────────────────────
BG = "#0f172a"
STATES = {
    "idle":        ("#22c55e", "#64748b", "#1e3a5f"),
    "recording":   ("#ef4444", "#ef4444", "#7f1d1d"),
    "translating": ("#f59e0b", "#f59e0b", "#78350f"),
    "playing":     ("#a855f7", "#a855f7", "#3b0764"),
}
LABELS = {
    "idle":        "Ready  ·  hold right ⌥ to record",
    "recording":   "Recording & translating…",
    "translating": "Finishing…",
    "playing":     "Playing translation…",
}

_BAR_TOP = 28
_BAR_BOT = 70
WIN_H    = _BAR_BOT + TEXT_H + 10

URL = "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"


class OpenAITranslate:
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

        self.state       = "idle"
        self._recording  = False
        self._rms        = 0.0
        self._levels     = [0.0] * BAR_N
        self._phase      = 0.0
        self._text       = ""
        self._audio_out  = []        # current recording's translated audio for replay
        self._kb         = KBController()
        self._api_key    = api_key

        # Persistent connection: one WebSocket for the whole app lifetime, so the
        # API's dynamic voice adaptation stays consistent and we skip the per-
        # recording handshake. Recordings are delimited by a generation counter.
        self._tx_q: queue.Queue = queue.Queue()   # mic → API sender
        self._rx_q: queue.Queue = queue.Queue()   # API audio → playback, holds (gen, chunk)
        self._gen          = 0       # bumped per recording; stale rx chunks are skipped
        self._done_gen     = 0       # highest gen whose playback has fully finished
        self._connected    = False   # True while the WebSocket session is live
        self._quitting     = False   # set on window close to unwind background loops
        self._last_rx      = 0.0     # time of last audio chunk received or played
        self._release_time = 0.0     # time the hotkey was released

        self._build_ui()
        self._start_audio()
        self._start_hotkey()
        self._start_connection()
        threading.Thread(target=self._playback_thread, daemon=True).start()
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
        self._dot = cv.create_oval(PAD, 11, PAD + 9, 20, fill=STATES["idle"][0], outline="")
        self._lbl = cv.create_text(PAD + 16, 15, text=LABELS["idle"],
                                   fill=STATES["idle"][1], font=("Helvetica Neue", 11), anchor="w")
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

        cv.create_line(PAD, _BAR_BOT + 8, WIN_W - PAD, _BAR_BOT + 8, fill="#1e293b", width=1)

        # Translated text — click to replay
        self._txt = cv.create_text(
            PAD, _BAR_BOT + 14, text="", fill="#475569",
            font=("Helvetica Neue", 11), anchor="nw", width=WIN_W - 2 * PAD,
            tags="replay",
        )
        cv.tag_bind("replay", "<Button-1>", lambda _: self._replay())
        cv.tag_bind("replay", "<Enter>",    lambda _: cv.itemconfig("replay", fill="#93c5fd"))
        cv.tag_bind("replay", "<Leave>",    lambda _: cv.config())  # reset in tick

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
        if self._recording and self._connected:
            chunk_i16 = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
            self._tx_q.put(chunk_i16)

    # ── Hotkey ─────────────────────────────────────────────────────────────────

    def _start_hotkey(self):
        def on_press(key):
            if key == HOTKEY and self.state in ("idle", "translating", "playing"):
                # New recording: bump generation so any in-flight audio/text from
                # the previous utterance is treated as stale, and clear buffers.
                self._gen        += 1
                self._text        = ""
                self._audio_out   = []
                self._recording   = True
                self.state        = "recording"

        def on_release(key):
            if key == HOTKEY and self._recording:
                self._recording    = False
                self._release_time = time.time()
                self.state         = "translating"

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    # ── Persistent translation connection ──────────────────────────────────────

    def _start_connection(self):
        threading.Thread(target=lambda: asyncio.run(self._connection_loop()),
                         daemon=True).start()

    async def _connection_loop(self):
        """Hold one WebSocket open for the app's lifetime; reconnect on drop."""
        while not self._quitting:
            try:
                await self._run_session()
            except Exception as e:
                if not self._quitting:
                    print(f"[openai_translate] connection lost: {e} — reconnecting in 2s")
                    self._connected = False
                    await asyncio.sleep(2)

    async def _run_session(self):
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with websockets.connect(URL, additional_headers=headers) as ws:
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {"audio": {"output": {"language": TARGET_LANG}}}
            }))
            # Drop any mic audio buffered while disconnected.
            while not self._tx_q.empty():
                self._tx_q.get_nowait()
            self._connected = True
            try:
                sender   = asyncio.create_task(self._sender(ws))
                receiver = asyncio.create_task(self._receiver(ws))
                done, pending = await asyncio.wait(
                    {sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    task.result()   # re-raise if a task failed
            finally:
                self._connected = False

    async def _sender(self, ws):
        """Stream mic chunks continuously; light silence keepalive while idle."""
        silence = np.zeros(2400, dtype=np.int16)   # 100 ms @ 24 kHz
        last_keepalive = time.time()
        while not self._quitting:
            sent = False
            try:
                while True:
                    chunk = self._tx_q.get_nowait()
                    await ws.send(json.dumps({
                        "type": "session.input_audio_buffer.append",
                        "audio": base64.b64encode(chunk.tobytes()).decode(),
                    }))
                    sent = True
            except queue.Empty:
                pass
            if not sent and not self._recording and time.time() - last_keepalive > 5:
                await ws.send(json.dumps({
                    "type": "session.input_audio_buffer.append",
                    "audio": base64.b64encode(silence.tobytes()).decode(),
                }))
                last_keepalive = time.time()
            await asyncio.sleep(0.005)

    async def _receiver(self, ws):
        """Route translation deltas to the current recording's buffers."""
        async for message in ws:
            if self._quitting:
                break
            event = json.loads(message)
            t = event.get("type", "")
            active = self.state in ("recording", "translating", "playing")
            if t == "session.output_transcript.delta" and active:
                delta = event.get("delta", "")
                self._text += delta
                try:
                    self._kb.type(delta)
                except Exception:
                    pass
            elif t == "session.output_audio.delta" and active:
                chunk = np.frombuffer(base64.b64decode(event["delta"]), dtype=np.int16)
                self._audio_out.append(chunk.copy())
                self._rx_q.put((self._gen, chunk))
                self._last_rx = time.time()
            elif t == "error":
                print(f"[event] error: {json.dumps(event)[:200]}")

    # ── Audio playback ─────────────────────────────────────────────────────────

    def _playback_thread(self):
        """Persistent: play audio for the current recording once the key is released."""
        with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            while not self._quitting:
                # Hold playback while still recording — the user doesn't want to
                # hear the translation while they're still speaking.
                if self.state == "recording":
                    time.sleep(0.02)
                    continue
                try:
                    gen, chunk = self._rx_q.get(timeout=0.1)
                except queue.Empty:
                    self._maybe_finish()
                    continue
                if gen != self._gen or self._done_gen >= self._gen:
                    continue   # stale chunk from a finished or superseded recording
                if self.state == "translating":
                    self.state = "playing"
                stream.write((chunk.astype(np.float32) / 32767.0).reshape(-1, 1))
                self._last_rx = time.time()

    def _maybe_finish(self):
        """Transition to idle once a recording's translation has fully drained."""
        if self.state == "playing" and time.time() - self._last_rx > 0.8:
            self._finish()
        elif self.state == "translating" and time.time() - self._release_time > 4.0:
            self._finish()

    def _finish(self):
        self._done_gen = self._gen
        if self._text.strip():
            pyperclip.copy(self._text.strip())
        self.state = "idle"

    def _replay(self):
        if not self._audio_out or self.state != "idle":
            return
        # Treat replay like a fresh playback generation so the playback thread
        # accepts the chunks again.
        self._gen     += 1
        self._last_rx  = time.time()
        self.state     = "playing"
        for chunk in list(self._audio_out):
            self._rx_q.put((self._gen, chunk))

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

        if self._text:
            text_col = "#e2e8f0" if state != "idle" else "#64748b"
            self.cv.itemconfig(self._txt, text=self._text.strip(), fill=text_col)

        self.root.after(1000 // FPS, self._tick)

    # ── Quit ───────────────────────────────────────────────────────────────────

    def _quit(self):
        self._quitting = True
        self._stream.stop()
        self._listener.stop()
        self.root.destroy()


if __name__ == "__main__":
    OpenAITranslate()
