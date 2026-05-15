#!/usr/bin/env python3
"""
Benchmark mlx_whisper against SuperWhisper on all recordings in ./recordings/.
Results are cached so reruns skip already-transcribed files.

Usage:
    python benchmark.py                              # default models
    python benchmark.py --model mlx-community/whisper-large-v3
    python benchmark.py --model mlx-community/whisper-tiny --model mlx-community/whisper-large-v3-turbo
"""

import argparse
import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path

RECORDINGS_DIR = Path(__file__).parent / "recordings"
CACHE_FILE = Path(__file__).parent / "benchmark_cache.json"
REPORT_FILE = Path(__file__).parent / "benchmark_report.html"
DEFAULT_MODELS = [
    "mlx-community/whisper-large-v3-turbo",
    "mlx-community/whisper-tiny",
]


def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def get_candidates() -> list[dict]:
    candidates = []
    for folder in sorted(RECORDINGS_DIR.iterdir()):
        wav = folder / "output.wav"
        meta_path = folder / "meta.json"
        if not wav.exists() or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        raw = meta.get("rawResult", "").strip()
        if not raw:
            continue
        candidates.append({
            "folder": folder.name,
            "wav": str(wav),
            "datetime": meta.get("datetime", ""),
            "duration_ms": meta.get("duration", 0),
            "sw_model": meta.get("modelName", meta.get("modelKey", "")),
            "sw_raw": meta.get("rawResult", "").strip(),
            "sw_result": meta.get("result", "").strip(),
            "sw_llm_result": meta.get("llmResult", "").strip() if meta.get("llmResult") else "",
            "sw_llm_model": meta.get("languageModelName", ""),
            "language": meta.get("languageSelected", ""),
            "mode": meta.get("modeName", ""),
        })
    return candidates


def transcribe_with_mlx(wav_path: str, model: str) -> tuple[str, float]:
    import mlx_whisper
    t0 = time.time()
    result = mlx_whisper.transcribe(wav_path, path_or_hf_repo=model, verbose=False)
    elapsed = time.time() - t0
    return result.get("text", "").strip(), round(elapsed, 2)


def audio_data_uri(wav_path: str) -> str:
    with open(wav_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:audio/wav;base64,{data}"


def word_count(text: str) -> int:
    return len(text.split()) if text.strip() else 0


def diff_class(sw_wc: int, mlx_wc: int) -> str:
    if sw_wc == 0:
        return ""
    ratio = mlx_wc / sw_wc
    if ratio < 0.5 or ratio > 2.0:
        return "big-diff"
    if ratio < 0.75 or ratio > 1.33:
        return "med-diff"
    return "ok"


def build_html(rows: list[dict], models: list[str]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    short_names = [m.split("/")[-1] for m in models]

    # Build per-model header columns
    model_headers = ""
    for name in short_names:
        model_headers += f'  <th>mlx: {name}</th>\n  <th>words</th>\n  <th>time</th>\n  <th>RT×</th>\n'

    table_rows = []
    for i, r in enumerate(rows, 1):
        audio_uri = audio_data_uri(r["wav"])
        sw_wc = word_count(r["sw_raw"])
        llm_col = f'<td class="text-cell">{r["sw_llm_result"] or "<em>—</em>"}</td>'

        model_cols = ""
        for m in models:
            key = f"mlx_{m}"
            mlx_text = r.get(key + "_text", "<em>—</em>")
            mlx_elapsed = r.get(key + "_elapsed", 0)
            mlx_wc = word_count(mlx_text)
            dc = diff_class(sw_wc, mlx_wc)
            rt_ratio = round(mlx_elapsed / (r["duration_ms"] / 1000), 2) if r["duration_ms"] and mlx_elapsed else "?"
            model_cols += f'<td class="text-cell mlx">{mlx_text}</td>'
            model_cols += f'<td class="{dc}">{mlx_wc}</td>'
            model_cols += f'<td>{mlx_elapsed}s</td>'
            model_cols += f'<td class="{dc}">{rt_ratio}×</td>'

        row = f"""
        <tr>
          <td class="num">{i}</td>
          <td class="date">{r['datetime'][:16]}</td>
          <td>{round(r['duration_ms']/1000, 1)}s</td>
          <td class="model-cell">{r['sw_model']}</td>
          <td class="mode">{r['mode']}</td>
          <td class="lang">{r['language']}</td>
          <td class="text-cell sw">{r['sw_raw']}</td>
          {llm_col}
          <td class="ok">{sw_wc}</td>
          {model_cols}
          <td><audio controls preload="none" src="{audio_uri}"></audio></td>
        </tr>"""
        table_rows.append(row)

    rows_html = "\n".join(table_rows)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SuperWhisper vs mlx_whisper — {now}</title>
<style>
  body {{ font-family: system-ui, sans-serif; font-size: 13px; background: #f5f5f5; margin: 0; padding: 16px; }}
  h1 {{ font-size: 18px; margin-bottom: 4px; }}
  .meta {{ color: #666; margin-bottom: 16px; font-size: 12px; }}
  table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 1px 3px rgba(0,0,0,.1); border-radius: 6px; overflow: hidden; }}
  th {{ background: #1a1a2e; color: white; padding: 8px 10px; text-align: left; white-space: nowrap; position: sticky; top: 0; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top; }}
  tr:hover td {{ background: #f0f4ff; }}
  .num {{ color: #999; width: 28px; }}
  .date {{ white-space: nowrap; color: #555; }}
  .text-cell {{ max-width: 260px; }}
  .text-cell.sw {{ background: #fafff8; }}
  .text-cell.mlx {{ background: #f8f8ff; }}
  .model-cell {{ font-size: 11px; color: #444; }}
  .mode {{ font-size: 11px; }}
  .lang {{ text-align: center; }}
  .ok {{ color: #2a7; }}
  .med-diff {{ color: #b80; font-weight: bold; }}
  .big-diff {{ color: #c00; font-weight: bold; }}
  audio {{ width: 200px; height: 32px; }}
  .legend {{ margin-top: 12px; font-size: 11px; color: #666; }}
  .legend span {{ display: inline-block; margin-right: 16px; }}
</style>
</head>
<body>
<h1>SuperWhisper vs mlx_whisper</h1>
<div class="meta">
  Generated: {now} &nbsp;|&nbsp;
  Recordings: {len(rows)} &nbsp;|&nbsp;
  Models: <strong>{", ".join(short_names)}</strong>
</div>
<table>
<thead>
<tr>
  <th>#</th>
  <th>Date</th>
  <th>Dur</th>
  <th>SW Model</th>
  <th>Mode</th>
  <th>Lang</th>
  <th>SuperWhisper (raw)</th>
  <th>SuperWhisper (LLM)</th>
  <th>SW words</th>
  {model_headers}
  <th>Audio</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
<div class="legend">
  <span><strong>RT ratio</strong>: mlx processing time ÷ audio duration (lower = faster than realtime)</span>
  <span class="ok">&#9632; word counts within 25%</span>
  <span class="med-diff">&#9632; 25–50% diff</span>
  <span class="big-diff">&#9632; >50% diff</span>
</div>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", dest="models", metavar="HF_REPO",
                        help="mlx_whisper HF model repo (can be repeated for multi-model comparison)")
    args = parser.parse_args()
    models = args.models or DEFAULT_MODELS

    print(f"Scanning {RECORDINGS_DIR} ...")
    candidates = get_candidates()
    print(f"Found {len(candidates)} recordings with content and audio.")
    print(f"Models: {models}")

    cache = load_cache()

    # Collect results for all models
    rows = []
    for i, rec in enumerate(candidates, 1):
        folder = rec["folder"]
        row = dict(rec)
        for model in models:
            cached_model = cache.get(model, {})
            key = f"mlx_{model}"
            if folder in cached_model:
                row[key + "_text"] = cached_model[folder]["mlx_text"]
                row[key + "_elapsed"] = cached_model[folder]["mlx_elapsed"]
                print(f"  [{i}/{len(candidates)}] {folder} [{model.split('/')[-1]}] (cached)")
            else:
                print(f"  [{i}/{len(candidates)}] {folder} [{model.split('/')[-1]}] ...", end=" ", flush=True)
                try:
                    mlx_text, mlx_elapsed = transcribe_with_mlx(rec["wav"], model)
                    print(f"{mlx_elapsed}s — {mlx_text[:60]!r}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    mlx_text, mlx_elapsed = f"[ERROR: {e}]", 0.0
                if model not in cache:
                    cache[model] = {}
                cache[model][folder] = {"mlx_text": mlx_text, "mlx_elapsed": mlx_elapsed}
                save_cache(cache)
                row[key + "_text"] = mlx_text
                row[key + "_elapsed"] = mlx_elapsed
        rows.append(row)

    rows.sort(key=lambda r: r["datetime"], reverse=True)

    print(f"\nBuilding report ({len(rows)} rows) ...")
    html = build_html(rows, models)
    REPORT_FILE.write_text(html, encoding="utf-8")
    print(f"Report saved to: {REPORT_FILE}")
    os.system(f'open "{REPORT_FILE}"')


if __name__ == "__main__":
    main()
