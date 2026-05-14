# mywhisper

A lightweight voice-to-text tool using [mlx_whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper), tailored to my needs. Hold a hotkey, speak, and get the transcription pasted at the cursor position.

## Setup

```bash
uv sync
source .venv/bin/activate
```

## benchmark.py

Internal script for evaluating mlx_whisper transcription quality against a set of reference recordings (from SuperWhisper). Not intended for general use — the `recordings/` directory is local-only and not checked in.

```bash
python benchmark.py                                        # default model
python benchmark.py --model mlx-community/whisper-large-v3
```

Generates `benchmark_report.html` with a side-by-side comparison of SuperWhisper and mlx_whisper transcripts, including inline audio playback.
