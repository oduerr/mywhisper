#!/usr/bin/env python3
"""Quick test: send a WAV file to gpt-realtime-translate and play back + print result."""

import asyncio, base64, json, os, sys
import numpy as np
import sounddevice as sd
import soundfile as sf
import websockets

TARGET_LANG = sys.argv[2] if len(sys.argv) > 2 else "zh"
WAV_FILE    = sys.argv[1] if len(sys.argv) > 1 else "recordings/1769077376/output.wav"
SAMPLE_RATE = 24000

async def run():
    # Load and resample to 24kHz PCM16
    audio_f32, sr = sf.read(WAV_FILE, dtype="float32")
    if audio_f32.ndim > 1:
        audio_f32 = audio_f32[:, 0]
    if sr != SAMPLE_RATE:
        import scipy.signal
        audio_f32 = scipy.signal.resample(audio_f32, int(len(audio_f32) * SAMPLE_RATE / sr))
    audio_i16 = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)
    audio_b64 = base64.b64encode(audio_i16.tobytes()).decode()
    print(f"Input:  {WAV_FILE}  ({len(audio_f32)/SAMPLE_RATE:.2f}s, resampled from {sr}Hz)")
    print(f"Target: {TARGET_LANG}")

    url     = "wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate"
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}

    text_parts, audio_parts = [], []

    async with websockets.connect(url, additional_headers=headers) as ws:
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {"audio": {"output": {"language": TARGET_LANG}}}
        }))
        await ws.send(json.dumps({"type": "session.input_audio_buffer.append", "audio": audio_b64}))
        await ws.send(json.dumps({"type": "session.close"}))
        print("Sent — waiting for response...")

        async for message in ws:
            event = json.loads(message)
            t = event.get("type", "")
            if t == "session.output_transcript.delta":
                text_parts.append(event.get("delta", ""))
                print(f"  text: {''.join(text_parts)}", end="\r")
            elif t == "session.output_audio.delta":
                chunk = np.frombuffer(base64.b64decode(event["delta"]), dtype=np.int16)
                audio_parts.append(chunk)
            elif t == "error":
                print(f"\n  error: {event['error']['message']}")
            elif t == "session.closed":
                break

    text = "".join(text_parts).strip()
    print(f"\nTranslation: {text}")

    if audio_parts:
        audio_out = np.concatenate(audio_parts).astype(np.float32) / 32767.0
        print(f"Playing {len(audio_out)/SAMPLE_RATE:.2f}s of audio...")
        sd.play(audio_out, samplerate=SAMPLE_RATE)
        sd.wait()
        print("Done.")
    else:
        print("No audio received.")

asyncio.run(run())
