# mywhisper

A lightweight voice-to-text tool using [mlx_whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), tailored to my needs. Hold a hotkey, speak, and get the transcription pasted at the cursor position.

## Setup

```bash
uv sync
source .venv/bin/activate
```

## Running the app

```bash
python mywhisper.py
```

A small floating window appears in the bottom-right corner. Hold **right ⌥ (Option)** to record, release to transcribe — the text is pasted at the cursor position in whatever app is focused.

First launch downloads and warm-starts the model (~15s); subsequent launches are fast.

### macOS permissions (required once)

The app needs two permissions:

**1. Accessibility** — for the global hotkey and simulated paste (Cmd+V):

- Open **System Settings → Privacy & Security → Accessibility**
- Click **+** and add your terminal app (Terminal.app or iTerm2)
- Restart the app

Without this you'll see `This process is not trusted!` and the hotkey won't work.

**2. Microphone** — macOS will prompt automatically on first use.

## benchmark.py

Internal script for evaluating mlx_whisper transcription quality against a set of reference recordings (from SuperWhisper). Not intended for general use — the `recordings/` directory is local-only and not checked in.

```bash
python benchmark.py                                        # default model
python benchmark.py --model mlx-community/whisper-large-v3
```

Generates `benchmark_report.html` with a side-by-side comparison of SuperWhisper and mlx_whisper transcripts, including inline audio playback.
