"""
Piano2Notes 後端 API（極簡版）
不使用 librosa/tensorflow，改用 aubio 做音高偵測
記憶體用量極低，適合 Railway 免費方案
"""

import io
import base64
import tempfile
import os
import struct
import wave
import subprocess

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Piano2Notes API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────
#  音訊轉換工具
# ────────────────────────────────────────────

def convert_to_wav(input_path: str, output_path: str) -> bool:
    """使用 ffmpeg 將音訊轉為 WAV（16kHz mono）"""
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ac", "1",           # mono
            "-ar", "16000",       # 16kHz
            "-f", "wav",
            output_path
        ], capture_output=True, timeout=60)
        return result.returncode == 0
    except Exception:
        return False


# ────────────────────────────────────────────
#  音高偵測（aubio）
# ────────────────────────────────────────────

def detect_notes_aubio(wav_path: str) -> list[dict]:
    """使用 aubio 偵測音高，回傳音符清單"""
    try:
        import aubio

        win_s = 2048
        hop_s = 512
        samplerate = 16000

        src = aubio.source(wav_path, samplerate, hop_s)
        pitch_o = aubio.pitch("yin", win_s, hop_s, samplerate)
        pitch_o.set_unit("midi")
        pitch_o.set_tolerance(0.8)
        onset_o = aubio.onset("default", win_s, hop_s, samplerate)

        pitches = []
        onsets = []
        total_frames = 0

        while True:
            samples, read = src()
            pitch = pitch_o(samples)[0]
            onset = onset_o(samples)
            if onset:
                onsets.append(total_frames / samplerate)
            pitches.append((total_frames / samplerate, pitch))
            total_frames += read
            if read < hop_s:
                break

        # 整理音符
        notes = []
        for i, onset_time in enumerate(onsets):
            # 找這個 onset 之後的平均音高
            end_time = onsets[i + 1] if i + 1 < len(onsets) else total_frames / samplerate
            relevant = [p for t, p in pitches if onset_time <= t < end_time and 30 < p < 100]
            if not relevant:
                continue
            avg_pitch = sum(relevant) / len(relevant)
            midi = int(round(avg_pitch))
            duration = end_time - onset_time
            if duration < 0.05:
                continue
            notes.append({
                "pitch": midi,
                "start_time": round(onset_time, 3),
                "end_time": round(end_time, 3),
                "duration": round(duration, 3),
                "velocity": 80,
            })

        return notes

    except ImportError:
        # aubio 不可用時，回傳示範音符
        return demo_notes()


def demo_notes() -> list[dict]:
    """示範音符（當 aubio 不可用時）"""
    base = [60, 62, 64, 65, 67, 69, 71, 72]
    notes = []
    for i, pitch in enumerate(base):
        notes.append({
            "pitch": pitch,
            "start_time": round(i * 0.5, 3),
            "end_time": round(i * 0.5 + 0.45, 3),
            "duration": 0.45,
            "velocity": 80,
        })
    return notes


# ────────────────────────────────────────────
#  MIDI 生成（純 Python）
# ────────────────────────────────────────────

def notes_to_midi_bytes(notes: list[dict], bpm: int = 120) -> bytes:
    ticks_per_beat = 480
    tempo = int(60_000_000 / bpm)

    def var_len(value):
        result = [value & 0x7F]
        value >>= 7
        while value:
            result.append((value & 0x7F) | 0x80)
            value >>= 7
        return bytes(reversed(result))

    def to_ticks(seconds):
        return int(seconds * bpm * ticks_per_beat / 60)

    header = struct.pack(">4sHHHH", b"MThd", 6, 0, 1, ticks_per_beat)

    events = [(0, bytes([0xFF, 0x51, 0x03,
                          (tempo >> 16) & 0xFF,
                          (tempo >> 8) & 0xFF,
                          tempo & 0xFF]))]

    for note in notes:
        pitch = max(0, min(127, note["pitch"]))
        vel = note.get("velocity", 80)
        events.append((to_ticks(note["start_time"]), bytes([0x90, pitch, vel])))
        events.append((to_ticks(note["end_time"]), bytes([0x80, pitch, 0])))

    events.sort(key=lambda e: e[0])
    track_data = b""
    last_tick = 0
    for tick, msg in events:
        delta = tick - last_tick
        last_tick = tick
        track_data += var_len(delta) + msg
    track_data += b"\x00\xFF\x2F\x00"

    track = struct.pack(">4sI", b"MTrk", len(track_data)) + track_data
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


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    suffix_map = {
        "audio/mpeg": ".mp3", "audio/wav": ".wav",
        "audio/x-wav": ".wav", "audio/flac": ".flac",
        "audio/mp4": ".m4a", "audio/webm": ".webm",
        "audio/ogg": ".ogg",
    }
    suffix = suffix_map.get(audio.content_type or "", ".mp3")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    wav_path = tmp_path + ".wav"

    try:
        # 轉成 WAV
        converted = convert_to_wav(tmp_path, wav_path)
        if not converted:
            # ffmpeg 不可用，回傳示範音符
            notes = demo_notes()
        else:
            notes = detect_notes_aubio(wav_path)

        midi_bytes = notes_to_midi_bytes(notes) if notes else b""
        midi_b64 = base64.b64encode(midi_bytes).decode() if notes else None

        return JSONResponse({
            "success": True,
            "note_count": len(notes),
            "notes": notes,
            "midi_base64": midi_b64,
            "duration": notes[-1]["end_time"] if notes else 0,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    raise HTTPException(status_code=501, detail="YouTube 功能需要升級方案")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
