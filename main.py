"""
Piano2Notes v6 - 快速版本
用零交叉率 + 能量峰值做快速音符偵測
"""
import base64, tempfile, os, struct, math, wave
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

app = FastAPI(title="Piano2Notes API", version="6.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def cors(request: Request, call_next):
    if request.method == "OPTIONS":
        return Response(status_code=200, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"})
    r = await call_next(request)
    r.headers["Access-Control-Allow-Origin"] = "*"
    return r

def mp3_to_wav(mp3_path, wav_path):
    from pydub import AudioSegment
    audio = AudioSegment.from_file(mp3_path)
    audio = audio.set_frame_rate(11025).set_channels(1).set_sample_width(2)
    audio.export(wav_path, format="wav")

def read_wav(wav_path):
    with wave.open(wav_path, 'rb') as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        samples = [struct.unpack_from('<h', raw, i)[0] / 32768.0
                   for i in range(0, len(raw) - 1, 2)]
    return samples, sr

def fast_pitch(frame, sr):
    """快速音高偵測：用自相關但只取關鍵 lag"""
    n = len(frame)
    energy = sum(s * s for s in frame) / n
    if energy < 0.001:
        return None
    
    mx = max(abs(s) for s in frame) or 1.0
    norm = [s / mx for s in frame]
    
    # 只搜尋鋼琴常見音域 C3-C6 (130-1047 Hz)
    min_lag = max(1, int(sr / 1047))
    max_lag = min(int(sr / 130), n // 2)
    
    best_lag, best_corr = min_lag, -1.0
    # 每隔 2 個 lag 取樣（速度加倍）
    for lag in range(min_lag, max_lag + 1, 2):
        corr = sum(norm[i] * norm[i + lag] for i in range(0, n - lag, 4))  # 每隔4取樣
        if corr > best_corr:
            best_corr, best_lag = corr, lag
    
    return sr / best_lag if best_corr >= 0.25 else None

def hz_to_midi(hz):
    if not hz or hz <= 0: return None
    m = int(round(69 + 12 * math.log2(hz / 440.0)))
    return m if 21 <= m <= 108 else None

def detect_notes(wav_path):
    samples, sr = read_wav(wav_path)
    # 使用較大的 hop_size 減少計算量
    frame_size = 1024
    hop_size = 512
    notes, current, history = [], None, []
    
    total = (len(samples) - frame_size) // hop_size
    for i in range(total):
        start = i * hop_size
        t = start / sr
        frame = samples[start:start + frame_size]
        midi = hz_to_midi(fast_pitch(frame, sr))
        
        if midi:
            history = (history + [midi])[-5:]
            stable = sorted(history)[len(history) // 2]
            if current is None:
                current = {"pitch": stable, "start_time": round(t, 3), "velocity": 80}
            elif abs(stable - current["pitch"]) > 2:
                current["end_time"] = round(t, 3)
                current["duration"] = round(t - current["start_time"], 3)
                if current["duration"] >= 0.1:
                    notes.append(current)
                current = {"pitch": stable, "start_time": round(t, 3), "velocity": 80}
        else:
            history = []
            if current:
                current["end_time"] = round(t, 3)
                current["duration"] = round(t - current["start_time"], 3)
                if current["duration"] >= 0.1:
                    notes.append(current)
                current = None
    
    if current:
        end_t = len(samples) / sr
        current.update(end_time=round(end_t, 3),
                      duration=round(end_t - current["start_time"], 3))
        if current["duration"] >= 0.1:
            notes.append(current)
    return notes

def to_midi(notes, bpm=120):
    ticks = 480
    tempo = int(60_000_000 / bpm)
    def vl(v):
        r = [v & 0x7F]; v >>= 7
        while v: r.append((v & 0x7F) | 0x80); v >>= 7
        return bytes(reversed(r))
    tt = lambda s: int(s * bpm * ticks / 60)
    hdr = struct.pack('>4sHHHH', b'MThd', 6, 0, 1, ticks)
    evts = [(0, bytes([0xFF,0x51,0x03,(tempo>>16)&0xFF,(tempo>>8)&0xFF,tempo&0xFF]))]
    for n in notes:
        p = max(0, min(127, n["pitch"]))
        evts += [(tt(n["start_time"]), bytes([0x90,p,80])),
                 (tt(n.get("end_time", n["start_time"]+0.5)), bytes([0x80,p,0]))]
    evts.sort(key=lambda e: e[0])
    td, lt = b'', 0
    for tick, msg in evts:
        td += vl(tick-lt)+msg; lt=tick
    td += b'\x00\xFF\x2F\x00'
    return hdr + struct.pack('>4sI', b'MTrk', len(td)) + td

@app.get("/")
def root(): return {"status": "ok", "version": "6.0.0"}

@app.get("/health")
def health(): return {"status": "healthy"}

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    suffix = {
        "audio/mpeg":".mp3","audio/wav":".wav","audio/x-wav":".wav",
        "audio/flac":".flac","audio/mp4":".m4a","audio/webm":".webm"
    }.get(audio.content_type, ".mp3")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    wav_path = tmp_path + ".wav"
    
    try:
        if suffix == ".wav":
            wav_path = tmp_path
        else:
            mp3_to_wav(tmp_path, wav_path)
        
        notes = detect_notes(wav_path)
        midi_b64 = base64.b64encode(to_midi(notes)).decode() if notes else None
        
        return JSONResponse(
            content={"success": True, "note_count": len(notes), "notes": notes,
                     "midi_base64": midi_b64,
                     "duration": notes[-1].get("end_time", 0) if notes else 0},
            headers={"Access-Control-Allow-Origin": "*"})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in set([tmp_path, wav_path]):
            try: os.unlink(p)
            except: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
