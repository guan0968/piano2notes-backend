"""
Piano2Notes 後端 API v3
改用自相關函數（Autocorrelation）偵測音高
比 FFT 更準確，記憶體用量同樣極小
"""

import base64
import tempfile
import os
import struct
import math
import wave
import subprocess

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

app = FastAPI(title="Piano2Notes API", version="3.0.0")

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


# ────────────────────────────────────────────
#  音訊轉換
# ────────────────────────────────────────────

def convert_to_wav(input_path: str, output_path: str):
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "22050",  # 提高取樣率以抓更高音
        "-ac", "1",
        "-f", "wav",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 失敗: {result.stderr.decode()}")


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


# ────────────────────────────────────────────
#  自相關音高偵測
# ────────────────────────────────────────────

def autocorrelate(frame: list[float], sr: int):
    """
    用自相關函數偵測基頻
    鋼琴音域：A0(27.5Hz) ~ C8(4186Hz)
    """
    n = len(frame)
    energy = sum(s * s for s in frame) / n
    if energy < 0.0005:  # 靜音門檻（降低以抓更小聲的音）
        return None

    # 正規化
    max_val = max(abs(s) for s in frame) or 1.0
    norm = [s / max_val for s in frame]

    # 計算自相關
    min_lag = int(sr / 4200)   # 最高音 C8
    max_lag = int(sr / 27.5)   # 最低音 A0
    max_lag = min(max_lag, n - 1)

    best_lag = min_lag
    best_corr = -1.0

    lag = min_lag
    while lag <= max_lag:
        corr = sum(norm[i] * norm[i + lag] for i in range(n - lag))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
        lag += 1

    # 相關性太低 → 非週期性聲音（噪音）
    if best_corr < 0.3:
        return None

    freq = sr / best_lag
    return freq


def hz_to_midi(hz):
    if not hz or hz <= 0:
        return None
    midi = 69 + 12 * math.log2(hz / 440.0)
    rounded = int(round(midi))
    if 21 <= rounded <= 108:  # A0 ~ C8
        return rounded
    return None


def detect_notes(wav_path: str) -> list[dict]:
    samples, sr = read_wav_samples(wav_path)

    frame_size = 2048   # 約 93ms @ 22050Hz
    hop_size = 256      # 約 12ms，比之前的 512 更密集

    notes = []
    current_note = None
    current_midi_history = []

    total_frames = (len(samples) - frame_size) // hop_size

    for i in range(total_frames):
        start = i * hop_size
        frame = samples[start:start + frame_size]
        t = start / sr

        freq = autocorrelate(frame, sr)
        midi = hz_to_midi(freq)

        if midi:
            current_midi_history.append(midi)
            # 用最近 3 個 frame 的中位數做穩定判斷
            if len(current_midi_history) > 3:
                current_midi_history.pop(0)
            stable_midi = sorted(current_midi_history)[len(current_midi_history) // 2]

            if current_note is None:
                current_note = {
                    "pitch": stable_midi,
                    "start_time": round(t, 3),
                    "velocity": min(100, int(sum(abs(s) for s in frame) / len(frame) * 500) + 50),
                }
            elif abs(stable_midi - current_note["pitch"]) > 1:
                # 音高改變，結束目前音符
                current_note["end_time"] = round(t, 3)
                current_note["duration"] = round(t - current_note["start_time"], 3)
                if current_note["duration"] >= 0.05:  # 最短 50ms
                    notes.append(current_note)
                current_note = {
                    "pitch": stable_midi,
                    "start_time": round(t, 3),
                    "velocity": min(100, int(sum(abs(s) for s in frame) / len(frame) * 500) + 50),
                }
        else:
            current_midi_history.clear()
            if current_note is not None:
                current_note["end_time"] = round(t, 3)
                current_note["duration"] = round(t - current_note["start_time"], 3)
                if current_note["duration"] >= 0.05:
                    notes.append(current_note)
                current_note = None

    # 收尾
    if current_note is not None:
        end_t = len(samples) / sr
        current_note["end_time"] = round(end_t, 3)
        current_note["duration"] = round(end_t - current_note["start_time"], 3)
        if current_note["duration"] >= 0.05:
            notes.append(current_note)

    return notes


# ────────────────────────────────────────────
#  MIDI 生成（純 Python）
# ────────────────────────────────────────────

def notes_to_midi_bytes(notes: list[dict], bpm: int = 120) -> bytes:
    ticks_per_beat = 480
    tempo = int(60_000_000 / bpm)

    def var_len(value):
        result = []
        result.append(value & 0x7F)
        value >>= 7
        while value:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        return bytes(reversed(result))

    def to_ticks(seconds):
        return int(seconds * bpm * ticks_per_beat / 60)

    header = struct.pack('>4sHHHH', b'MThd', 6, 0, 1, ticks_per_beat)
    events = [(0, bytes([0xFF, 0x51, 0x03,
                         (tempo >> 16) & 0xFF,
                         (tempo >> 8) & 0xFF,
                         tempo & 0xFF]))]

    for note in notes:
        pitch = max(0, min(127, note["pitch"]))
        vel = note.get("velocity", 80)
        events.append((to_ticks(note["start_time"]), bytes([0x90, pitch, vel])))
        events.append((to_ticks(note.get("end_time", note["start_time"] + 0.5)),
                       bytes([0x80, pitch, 0])))

    events.sort(key=lambda e: e[0])
    track_data = b''
    last_tick = 0
    for tick, msg in events:
        delta = tick - last_tick
        last_tick = tick
        track_data += var_len(delta) + msg
    track_data += b'\x00\xFF\x2F\x00'
    track = struct.pack('>4sI', b'MTrk', len(track_data)) + track_data
    return header + track


# ────────────────────────────────────────────
#  路由
# ────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Piano2Notes API v3 🎹"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    suffix_map = {
        "audio/mpeg": ".mp3", "audio/wav": ".wav",
        "audio/x-wav": ".wav", "audio/flac": ".flac",
        "audio/mp4": ".m4a", "audio/webm": ".webm",
        "audio/ogg": ".ogg",
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
            midi_bytes = notes_to_midi_bytes(notes)
            midi_b64 = base64.b64encode(midi_bytes).decode()

        return JSONResponse(
            content={
                "success": True,
                "note_count": len(notes),
                "notes": notes,
                "midi_base64": midi_b64,
                "duration": notes[-1].get("end_time", 0) if notes else 0,
            },
            headers={"Access-Control-Allow-Origin": "*"},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"轉譜失敗：{str(e)}")

    finally:
        for p in [tmp_path, wav_path]:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


class YoutubeRequest(BaseModel):
    url: str

@app.post("/transcribe-youtube")
async def transcribe_youtube(req: YoutubeRequest):
    return JSONResponse(
        content={"success": False, "detail": "YouTube 功能需要升級方案"},
        headers={"Access-Control-Allow-Origin": "*"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
