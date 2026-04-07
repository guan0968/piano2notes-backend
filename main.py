"""
Piano2Notes 後端 API
使用 Spotify Basic Pitch 將音訊轉換為 MIDI / 音符

安裝依賴：
  pip install fastapi uvicorn python-multipart basic-pitch mido

啟動：
  uvicorn main:app --reload --port 8000

部署到 Railway：
  1. 將此資料夾推上 GitHub
  2. 在 Railway 建立新專案 → 連接 GitHub repo
  3. Railway 自動偵測 Procfile 並部署
"""

import io
import base64
import tempfile
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Basic Pitch (Spotify)
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

# MIDI 處理
import mido

app = FastAPI(title="Piano2Notes API", version="1.0.0")

# 允許前端跨域請求（把你的 Vercel 網址加進去）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "https://*.vercel.app",
        # 加入你的正式網域，例如：
        # "https://piano2notes.tw",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────
#  工具函式
# ────────────────────────────────────────────

def midi_to_base64(midi_path: str) -> str:
    """將 MIDI 檔案轉為 base64 字串回傳給前端"""
    with open(midi_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_notes_from_midi(midi_path: str) -> list[dict]:
    """
    從 MIDI 檔案解析音符清單
    回傳格式: [{ pitch, start_time, end_time, velocity }, ...]
    """
    mid = mido.MidiFile(midi_path)
    notes = []
    tempo = 500000  # 預設 120 BPM
    ticks_per_beat = mid.ticks_per_beat

    for track in mid.tracks:
        current_time = 0
        active_notes = {}

        for msg in track:
            current_time += msg.time
            time_sec = mido.tick2second(current_time, ticks_per_beat, tempo)

            if msg.type == "set_tempo":
                tempo = msg.tempo

            elif msg.type == "note_on" and msg.velocity > 0:
                active_notes[msg.note] = {
                    "pitch": msg.note,
                    "start_time": round(time_sec, 3),
                    "velocity": msg.velocity,
                }

            elif msg.type in ("note_off",) or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                if msg.note in active_notes:
                    note = active_notes.pop(msg.note)
                    note["end_time"] = round(time_sec, 3)
                    note["duration"] = round(time_sec - note["start_time"], 3)
                    notes.append(note)

    # 依開始時間排序
    notes.sort(key=lambda n: n["start_time"])
    return notes


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
    """
    接收音訊檔案，使用 Basic Pitch 轉譜
    回傳：音符清單 + MIDI base64
    """
    # 驗證檔案類型
    allowed_types = {
        "audio/mpeg", "audio/wav", "audio/x-wav",
        "audio/flac", "audio/mp4", "audio/m4a",
        "audio/ogg", "audio/webm",
    }
    if audio.content_type and audio.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"不支援的音訊格式：{audio.content_type}")

    # 儲存到暫存檔
    suffix = Path(audio.filename or "audio.mp3").suffix or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # ── Basic Pitch 轉譜 ──
        # model_output: 原始模型輸出
        # midi_data:    pretty_midi 物件
        # note_events:  [(start, end, pitch, amplitude, pitch_bend)]
        model_output, midi_data, note_events = predict(
            tmp_path,
            ICASSP_2022_MODEL_PATH,
            # 可調整參數：
            onset_threshold=0.5,      # 音符起始偵測靈敏度 (0–1)
            frame_threshold=0.3,      # 音框偵測靈敏度
            minimum_note_length=58,   # 最短音符（毫秒）
            minimum_frequency=None,   # 最低頻率 (Hz)，None = 不限
            maximum_frequency=None,   # 最高頻率 (Hz)，None = 不限
            multiple_pitch_bends=False,
            melodia_trick=True,       # 改善旋律追蹤
        )

        # 將 pretty_midi 輸出存為 MIDI 檔
        midi_out_path = tmp_path.replace(suffix, "_out.mid")
        midi_data.write(midi_out_path)

        # 轉為 base64 回傳
        midi_b64 = midi_to_base64(midi_out_path)

        # 整理音符清單（前端顯示用）
        notes = []
        for start, end, pitch, amplitude, _ in note_events:
            notes.append({
                "pitch": int(pitch),
                "start_time": round(float(start), 3),
                "end_time": round(float(end), 3),
                "duration": round(float(end - start), 3),
                "velocity": int(amplitude * 127),
            })

        return JSONResponse({
            "success": True,
            "note_count": len(notes),
            "notes": notes,
            "midi_base64": midi_b64,
            "duration": round(midi_data.get_end_time(), 2),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"轉譜失敗：{str(e)}")

    finally:
        # 清理暫存檔
        for path in [tmp_path, tmp_path.replace(suffix, "_out.mid")]:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


class YoutubeRequest(BaseModel):
    url: str


@app.post("/transcribe-youtube")
async def transcribe_youtube(req: YoutubeRequest):
    """
    接收 YouTube URL，使用 yt-dlp 下載音訊後轉譜
    需要額外安裝：pip install yt-dlp
    """
    try:
        import yt_dlp  # noqa
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="yt-dlp 未安裝。請執行：pip install yt-dlp",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "audio")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_path,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
            }],
            "quiet": True,
        }

        try:
            import yt_dlp
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([req.url])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"無法下載影片：{str(e)}")

        mp3_path = out_path + ".mp3"
        if not os.path.exists(mp3_path):
            raise HTTPException(status_code=500, detail="音訊下載失敗")

        # 複用上面的轉譜邏輯
        model_output, midi_data, note_events = predict(mp3_path, ICASSP_2022_MODEL_PATH)

        midi_out = mp3_path.replace(".mp3", ".mid")
        midi_data.write(midi_out)
        midi_b64 = midi_to_base64(midi_out)

        notes = [
            {
                "pitch": int(p),
                "start_time": round(float(s), 3),
                "end_time": round(float(e), 3),
                "velocity": int(a * 127),
            }
            for s, e, p, a, _ in note_events
        ]

        return JSONResponse({
            "success": True,
            "note_count": len(notes),
            "notes": notes,
            "midi_base64": midi_b64,
        })


# ────────────────────────────────────────────
#  本地開發入口
# ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
