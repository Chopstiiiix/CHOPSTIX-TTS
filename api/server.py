from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch, torchaudio, io, os, subprocess
from pathlib import Path

app = FastAPI(title="Chopstix TTS API", version="1.0.0")

MODEL = None
VOICES_DIR = Path("voices")

class SynthesizeRequest(BaseModel):
    text: str
    voice_id: str = "DANIEL"
    speed: float = 1.0

@app.on_event("startup")
async def load_model():
    global MODEL
    print("Loading Chopstix TTS model...")
    from f5_tts.api import F5TTS
    MODEL = F5TTS(model="F5TTS_v1_Base")
    print("Model loaded successfully")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.get("/voices")
def list_voices():
    voices = [f.stem for f in VOICES_DIR.glob("*.wav")]
    return {"voices": sorted(voices)}

@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")
    ref_path = VOICES_DIR / f"{req.voice_id}.wav"
    if not ref_path.exists():
        raise HTTPException(404, f"Voice {req.voice_id} not found")
    wav, sr, _ = MODEL.infer(
        ref_file=str(ref_path),
        ref_text="",
        gen_text=req.text,
        file_wave="/tmp/out.wav"
    )
    result = subprocess.run([
        "ffmpeg", "-y", "-i", "/tmp/out.wav",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,equalizer=f=8000:width_type=o:width=2:g=-4,equalizer=f=12000:width_type=o:width=2:g=-3",
        "/tmp/out_clean.wav"
    ], capture_output=True)
    with open("/tmp/out_clean.wav", "rb") as f:
        buf = io.BytesIO(f.read())
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav",
        headers={"Content-Disposition": f"attachment; filename={req.voice_id}.wav"})

@app.post("/clone")
async def clone_voice(
    voice_name: str,
    reference_audio: UploadFile = File(...)
):
    ref_bytes = await reference_audio.read()
    ref_path = VOICES_DIR / f"{voice_name}.wav"
    with open(f"/tmp/{voice_name}_raw.wav", "wb") as f:
        f.write(ref_bytes)
    subprocess.run([
        "ffmpeg", "-y", "-i", f"/tmp/{voice_name}_raw.wav",
        "-ar", "24000", "-ac", "1", str(ref_path)
    ], capture_output=True)
    return {"status": "ok", "voice_id": voice_name}
