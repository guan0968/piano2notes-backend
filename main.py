"""
Piano2Notes 後端 API
使用 librosa 做音高偵測（不需要 tensorflow）
"""

import io
import base64
import tempfile
import os
import struct

import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Piano2Notes API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 部署後可改成你的 Vercel 網址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────
#  音高偵測（librosa）
# ────────────────────────────────────────────

def hz_to_midi(hz):
    """頻率轉 MIDI 音符編號"""
    if hz <= 0:
        return None
    midi = 69 + 12 * np.log2(hz / 440.0)
    return int(round(midi))


def detect_notes(audio_path: str) -> list[dict]:
    """
    使用 librosa 偵測音符
    回傳: [{ pitch, start_time, end_time, velocity }, ...]
    """
    # 載入音訊
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # 使用 pyin 演算法偵測音高（適合旋律）
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        frame_length=2048,
    )

    # 計算每個 frame 的時間
    hop_length = 512
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # 將連續相同音高的 frame 合併成音符
    notes = []
    current_note = None

    for i, (t, freq, voiced) in enumerate(zip(times, f0, voiced_flag)):
        if voiced and freq is not None and not np.isnan(freq):
            midi = hz_to_midi(freq)
            if midi is None:
                continue

            if current_note is None:
                current_note = {
                    "pitch": midi,
                    "start_time": round(float(t), 3),
                    "velocity": 80,
                }
            elif abs(midi - current_note["pitch"]) > 1:
                # 音高改變，結束目前音符
                current_note["end_time"] = round(float(t), 3)
                current_note["duration"] = round(
                    current_note["end_time"] - current_note["start_time"], 3
                )
                if current_note["duration"] > 0.05:
                    notes.append(current_note)
                current_note = {
                    "pitch": midi,
                    "start_time": round(float(t), 3),
                    "velocity": 80,
                }
        else:
            if current_note is not None:
                current_note["end_time"] = round(float(t), 3)
                current_note["duration"] = round(
                    current_note["end_time"] - current_note["start_time"], 3
                )
                if current_note["duration"] > 0.05:
                    notes.append(current_note)
                current_note = None

    return notes


# ────────────────────────────────────────────
#  MIDI 生成（純 Python，不需要額外套件）
# ────────────────────────────────────────────

def notes_to_midi_bytes(notes: list[dict], bpm: int = 120) -> bytes:
    """將音符清單轉為 MIDI 檔案的 bytes"""
    ticks_per_beat = 480
    tempo = int(60_000_000 / bpm)  # microseconds per beat

    def var_len(value):
        result = []
        result.append(value & 0x7F)
        value >>= 7
        while value:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        return bytes(reversed(result))

    def ms_to_ticks(seconds):
        return int(seconds * bpm * ticks_per_beat / 60)

    # Header chunk
    header = struct.pack('>4sHHHH', b'MThd', 6, 0, 1, ticks_per_beat)

    # Track events
    events = []
    # Tempo event
    events.append((0, bytes([0xFF, 0x51, 0x03,
                              (tempo >> 16) & 0xFF,
                              (tempo >> 8) & 0xFF,
                              tempo & 0xFF])))

    for note in notes:
        on_tick = ms_to_ticks(note["start_time"])
        off_tick = ms_to_ticks(note["end_time"])
        pitch = max(0, min(127, note["pitch"]))
        vel = note.get("velocity", 80)
        events.append((on_tick, bytes([0x90, pitch, vel])))
        events.append((off_tick, bytes([0x80, pitch, 0])))

    # Sort by tick, convert to delta times
    events.sort(key=lambda e: e[0])
    track_data = b''
    last_tick = 0
    for tick, msg in events:
        delta = tick - last_tick
        last_tick = tick
        track_data += var_len(delta) + msg

    # End of track
    track_data += b'\x00\xFF\x2F\x00'

    # Track chunk
    track = struct.pack('>4sI', b'MTrk', len(track_data)) + track_data

    return header + track


# ────────────────────────────────────────────
#  路由
# ────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "Piano2Notes API is running 🎹"}


@app.get("/health")
def health():
    return {"status": "healthy"}

from fastapi import Response
from fastapi.responses import JSONResponse

@app.options("/transcribe")
async def options_transcribe():
    return Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

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
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        notes = detect_notes(tmp_path)

        # 產生 MIDI
        if notes:
            midi_bytes = notes_to_midi_bytes(notes)
            midi_b64 = base64.b64encode(midi_bytes).decode("utf-8")
        else:
            midi_b64 = None

        duration = notes[-1]["end_time"] if notes else 0

        return JSONResponse({
            "success": True,
            "note_count": len(notes),
            "notes": notes,
            "midi_base64": midi_b64,
            "duration": duration,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"轉譜失敗：{str(e)}")

    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


class YoutubeRequest(BaseModel):
    url: str


@app.post("/transcribe-youtube")
async def transcribe_youtube(req: YoutubeRequest):
    try:
        import yt_dlp
    except ImportError:
        raise HTTPException(status_code=501, detail="yt-dlp 未安裝")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "audio")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_path,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
            "quiet": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([req.url])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"下載失敗：{str(e)}")

        mp3_path = out_path + ".mp3"
        if not os.path.exists(mp3_path):
            raise HTTPException(status_code=500, detail="音訊下載失敗")

        notes = detect_notes(mp3_path)
        midi_bytes = notes_to_midi_bytes(notes) if notes else b''
        midi_b64 = base64.b64encode(midi_bytes).decode("utf-8") if notes else None

        return JSONResponse({
            "success": True,
            "note_count": len(notes),
            "notes": notes,
            "midi_base64": midi_b64,
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
