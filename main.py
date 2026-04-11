"""
Piano2Notes 後端 API v4
自相關音高偵測 + 自動尋找 ffmpeg
"""

import base64
import tempfile
import os
import struct
import math
import wave
import subprocess
import glob
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

app = FastAPI(title="Piano2Notes API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_cors_header(request: Request, call_next):
    if request.method == "OPTIONS":
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            },
        )
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


def find_ffmpeg():
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    for p in glob.glob("/nix/store/*/bin/ffmpeg"):
        return p
    for p in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if os.path.exists(p):
            return p
    return None


def convert_to_wav(input_path: str, output_path: str):
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        subprocess.run(["apt-get", "install", "-y", "ffmpeg"],
                       capture_output=True, timeout=120)
        ffmpeg = find_ffmpeg() or "ffmpeg"

    cmd = [ffmpeg, "-y", "-i", input_path,
           "-ar", "22050", "-ac", "1", "-f", "wav", output_path]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr.decode()[:200]}")


def read_wav_samples(wav_path: str):
    with wave.open(wav_path, 'rb') as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        samples = [
            struct.unpack_from('<h', raw, i)[0] / 32768.0
            for i in range(0, min(len(raw), n_frames * 2), 2)
        ]
    return samples, sr


def autocorrelate(frame, sr):
    n = len(frame)
    energy = sum(s * s for s in frame) / n
    if energy < 0.0005:
        return None
    max_val = max(abs(s) for s in frame) or 1.0
    norm = [s / max_val for s in frame]
    min_lag = max(1, int(sr / 4200))
    max_lag = min(int(sr / 27.5), n - 1)
    best_lag, best_corr = min_lag, -1.0
    for lag in range(min_lag, max_lag + 1):
        corr = sum(norm[i] * norm[i + lag] for i in range(n - lag))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    if best_corr < 0.3:
        return None
    return sr / best_lag


def hz_to_midi(hz):
    if not hz or hz <= 0:
        return None
    midi = int(round(69 + 12 * math.log2(hz / 440.0)))
    return midi if 21 <= midi <= 108 else None


def detect_notes(wav_path: str):
    samples, sr = read_wav_samples(wav_path)
    frame_size, hop_size = 2048, 256
    notes, current_note, history = [], None, []

    for i in range((len(samples) - frame_size) // hop_size):
        start = i * hop_size
        frame = samples[start:start + frame_size]
        t = start / sr
        midi = hz_to_midi(autocorrelate(frame, sr))

        if midi:
            history.append(midi)
            if len(history) > 3:
                history.pop(0)
            stable = sorted(history)[len(history) // 2]
            if current_note is None:
                current_note = {"pitch": stable, "start_time": round(t, 3), "velocity": 80}
            elif abs(stable - current_note["pitch"]) > 1:
                current_note["end_time"] = round(t, 3)
                current_note["duration"] = round(t - current_note["start_time"], 3)
                if current_note["duration"] >= 0.05:
                    notes.append(current_note)
                current_note = {"pitch": stable, "start_time": round(t, 3), "velocity": 80}
        else:
            history.clear()
            if current_note:
                current_note["end_time"] = round(t, 3)
                current_note["duration"] = round(t - current_note["start_time"], 3)
                if current_note["duration"] >= 0.05:
                    notes.append(current_note)
                current_note = None

    if current_note:
        end_t = len(samples) / sr
        current_note["end_time"] = round(end_t, 3)
        current_note["duration"] = round(end_t - current_note["start_time"], 3)
        if current_note["duration"] >= 0.05:
            notes.append(current_note)
    return notes


def notes_to_midi_bytes(notes, bpm=120):
    ticks = 480
    tempo = int(60_000_000 / bpm)

    def vl(v):
        r = [v & 0x7F]
        v >>= 7
        while v:
            r.append((v & 0x7F) | 0x80)
            v >>= 7
        return bytes(reversed(r))

    def tt(s):
        return int(s * bpm * ticks / 60)

    hdr = struct.pack('>4sHHHH', b'MThd', 6, 0, 1, ticks)
    evts = [(0, bytes([0xFF, 0x51, 0x03,
                       (tempo >> 16) & 0xFF, (tempo >> 8) & 0xFF, tempo & 0xFF]))]
    for n in notes:
        p = max(0, min(127, n["pitch"]))
        evts.append((tt(n["start_time"]), bytes([0x90, p, n.get("velocity", 80)])))
        evts.append((tt(n.get("end_time", n["start_time"] + 0.5)), bytes([0x80, p, 0])))
    evts.sort(key=lambda e: e[0])
    td, lt = b'', 0
    for tick, msg in evts:
        td += vl(tick - lt) + msg
        lt = tick
    td += b'\x00\xFF\x2F\x00'
    return hdr + struct.pack('>4sI', b'MTrk', len(td)) + td


@app.get("/")
def root():
    ffmpeg = find_ffmpeg()
    return {"status": "ok", "ffmpeg": ffmpeg or "not found"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    suffix_map = {
        "audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav",
        "audio/flac": ".flac", "audio/mp4": ".m4a",
        "audio/webm": ".webm", "audio/ogg": ".ogg",
    }
    suffix = suffix_map.get(audio.content_type, ".mp3")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    wav_path = tmp_path + ".wav"

    try:
        convert_to_wav(tmp_path, wav_path)
        notes = detect_notes(wav_path)
        midi_b64 = None
        if notes:
            midi_b64 = base64.b64encode(notes_to_midi_bytes(notes)).decode()
        return JSONResponse(
            content={"success": True, "note_count": len(notes), "notes": notes,
                     "midi_base64": midi_b64,
                     "duration": notes[-1].get("end_time", 0) if notes else 0},
            headers={"Access-Control-Allow-Origin": "*"},
        )
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in [tmp_path, wav_path]:
            try:
                os.unlink(p)
            except Exception:
                pass


@app.post("/transcribe-youtube")
async def transcribe_youtube(req: BaseModel):
    return JSONResponse(
        content={"success": False, "detail": "YouTube 需要升級方案"},
        headers={"Access-Control-Allow-Origin": "*"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
