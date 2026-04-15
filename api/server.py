from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch, torchaudio, io, os, uuid
from pathlib import Path

app = FastAPI(title="Chopstix TTS API", version="1.0.0")

MODEL = None
VOICES_DIR = Path("api/voices")
VOICES_DIR.mkdir(exist_ok=True)

class SynthesizeRequest(BaseModel):
    text: str
    voice_id: str = "default"
    speed: float = 1.0

class CloneRequest(BaseModel):
    text: str
    voice_name: str

@app.on_event("startup")
async def load_model():
    global MODEL
    print("Loading Chopstix TTS model...")
    try:
        from f5_tts.api import F5TTS
        ckpt = os.getenv("MODEL_CKPT", "checkpoints/finetuned/model_best.pt")
        MODEL = F5TTS(model_type="F5TTS_v1_Base", ckpt_file=ckpt)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model load error: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.get("/voices")
def list_voices():
    voices = [f.stem for f in VOICES_DIR.glob("*.wav")]
    return {"voices": voices}

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
        speed=req.speed
    )
    buf = io.BytesIO()
    torchaudio.save(buf, wav, sr, format="wav")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")

@app.post("/clone")
async def clone_voice(
    text: str,
    voice_name: str,
    reference_audio: UploadFile = File(...)
):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")
    ref_bytes = await reference_audio.read()
    ref_path = VOICES_DIR / f"{voice_name}.wav"
    with open(ref_path, "wb") as f:
        f.write(ref_bytes)
    wav, sr, _ = MODEL.infer(
        ref_file=str(ref_path),
        ref_text="",
        gen_text=text
    )
    buf = io.BytesIO()
    torchaudio.save(buf, wav, sr, format="wav")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")
