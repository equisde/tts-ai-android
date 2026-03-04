import os
import io
import json
import numpy as np
import onnxruntime as ort
import librosa
import whisper
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from contextlib import asynccontextmanager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "base")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)

models = {"vocab": {}, "sess_pre": None, "sess_trans": None, "sess_dec": None, "whisper": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=== Iniciando Servidor TTS Profesional (Fallback System) ===")
    models["whisper"] = whisper.load_model("base")
    vocab_path = os.path.join(MODEL_DIR, "vocab.txt")
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f.readlines()):
                char = line.strip()
                if char: models["vocab"][char[0]] = idx
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(os.cpu_count() * 0.9))
        try:
            models["sess_pre"] = ort.InferenceSession(os.path.join(MODEL_DIR, "F5_Preprocess.onnx"), opts)
            models["sess_trans"] = ort.InferenceSession(os.path.join(MODEL_DIR, "F5_Transformer.onnx"), opts)
            models["sess_dec"] = ort.InferenceSession(os.path.join(MODEL_DIR, "F5_Decode.onnx"), opts)
            print("IA Lista.")
        except Exception as e: print(f"Error carga: {e}")
    yield

app = FastAPI(title="F5-TTS Voice Server", lifespan=lifespan)

def align_and_cast(session, input_name, data):
    for inp in session.get_inputs():
        if input_name in inp.name:
            if 'int16' in inp.type: data = (data * 32767).astype(np.int16)
            elif 'float' in inp.type: data = data.astype(np.float32)
            elif 'int64' in inp.type: data = data.astype(np.int64)
            elif 'int32' in inp.type: data = data.astype(np.int32)
            e_s = inp.shape
            while len(data.shape) < len(e_s): data = np.expand_dims(data, axis=0)
            a_s = data.shape
            if len(e_s) == len(a_s):
                for i in range(len(e_s)):
                    if isinstance(e_s[i], int) and e_s[i] > 0 and e_s[i] != a_s[i]:
                        for j in range(len(a_s)):
                            if a_s[j] == e_s[i]:
                                axes = list(range(len(a_s)))
                                axes[i], axes[j] = axes[j], axes[i]
                                data = np.transpose(data, axes)
                                break
            return data
    return data

@app.post("/clone")
async def clone(data: str = Form(...), audio: UploadFile = File(...)):
    req = json.loads(data)
    voice_id = req.get("voice_id", "voice")
    p_audio = os.path.join(PROFILES_DIR, f"{voice_id}.wav")
    p_text = os.path.join(PROFILES_DIR, f"{voice_id}.txt")
    with open(p_audio, "wb") as f: f.write(await audio.read())
    result = models["whisper"].transcribe(p_audio, language="es")
    ref_text = result["text"].strip()
    with open(p_text, "w", encoding="utf-8") as f: f.write(ref_text)
    return {"status": "success", "transcription": ref_text}

@app.post("/synthesize")
async def synthesize(data: str = Form(...)):
    req = json.loads(data)
    gen_text = req.get("text", "")
    voice_id = req.get("voice_id", "default")
    
    p_audio = os.path.join(PROFILES_DIR, f"{voice_id}.wav")
    p_text = os.path.join(PROFILES_DIR, f"{voice_id}.txt")

    # SISTEMA DE FALLBACK PARA VOCES BASE
    if not os.path.exists(p_audio):
        print(f"INFO: Usando voz base para {voice_id}")
        # Generar audio de referencia silencioso/neutro de 2s si no existe el clonado
        ref_audio = np.zeros(24000 * 2, dtype=np.float32)
        ref_text = ""
    else:
        ref_audio, _ = librosa.load(p_audio, sr=24000, mono=True)
        with open(p_text, "r", encoding="utf-8") as f: ref_text = f.read()

    tokens = np.array([models["vocab"].get(c, 0) for c in (ref_text + gen_text)], dtype=np.int64)
    max_dur = np.array([int(len(ref_audio)/256 + 1 + len(gen_text)*2)], dtype=np.int64)
    
    pre_in = {}
    for inp in models["sess_pre"].get_inputs():
        if 'audio' in inp.name: pre_in[inp.name] = align_and_cast(models["sess_pre"], 'audio', np.expand_dims(ref_audio, axis=(0,1)))
        elif 'text_ids' in inp.name: pre_in[inp.name] = align_and_cast(models["sess_pre"], 'text_ids', np.expand_dims(tokens, axis=0))
        elif 'max_duration' in inp.name: pre_in[inp.name] = align_and_cast(models["sess_pre"], 'max_duration', max_dur)

    pre_outs = models["sess_pre"].run(None, pre_in)
    noise = pre_outs[0]
    
    for step in range(30):
        trans_in = {}
        for i, inp in enumerate(models["sess_trans"].get_inputs()):
            if 'noise' in inp.name: trans_in[inp.name] = noise
            elif 'time_step' in inp.name: trans_in[inp.name] = align_and_cast(models["sess_trans"], 'time_step', np.array([step]))
            else:
                # Mapeo por índice para soportes
                trans_in[inp.name] = align_and_cast(models["sess_trans"], inp.name, pre_outs[i+1])
        noise = models["sess_trans"].run(None, trans_in)[0]
        
    audio_out = models["sess_dec"].run(None, {"denoised": noise, "ref_signal_len": pre_outs[7]})[0]
    pcm = (audio_out.flatten() * 32767).astype(np.int16).tobytes()
    return Response(content=pcm, media_type="audio/pcm")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)