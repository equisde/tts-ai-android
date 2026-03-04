import os
import io
import json
import numpy as np
import onnxruntime as ort
import librosa
import whisper
import struct
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
    print("=== Iniciando Servidor TTS (Audio Fix) ===")
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
            print(f"Modelos cargados exitosamente.")
        except Exception as e: print(f"Error carga: {e}")
    yield

app = FastAPI(title="F5-TTS Audio Fix Server", lifespan=lifespan)

def get_aligned_tensor(session, input_name, data):
    for inp in session.get_inputs():
        if input_name in inp.name:
            if 'int16' in inp.type: data = (data * 32767).astype(np.int16)
            elif 'float' in inp.type: data = data.astype(np.float32)
            elif 'int64' in inp.type: data = data.astype(np.int64)
            elif 'int32' in inp.type: data = data.astype(np.int32)
            
            e_shape = inp.shape
            a_shape = data.shape
            if len(e_shape) == len(a_shape):
                for i in range(len(e_shape)):
                    target = e_shape[i]
                    if isinstance(target, int) and target > 0 and target != a_shape[i]:
                        for j in range(len(a_shape)):
                            if a_shape[j] == target:
                                axes = list(range(len(a_shape)))
                                axes[i], axes[j] = axes[j], axes[i]
                                return np.transpose(data, axes)
            return data
    return data

@app.post("/clone")
async def clone(data: str = Form(...), audio: UploadFile = File(...)):
    req = json.loads(data)
    v_id = req.get("voice_id", "voice")
    p_audio = os.path.join(PROFILES_DIR, f"{v_id}.wav")
    p_text = os.path.join(PROFILES_DIR, f"{v_id}.txt")
    with open(p_audio, "wb") as f: f.write(await audio.read())
    result = models["whisper"].transcribe(p_audio, language="es")
    ref_text = result["text"].strip()
    with open(p_text, "w", encoding="utf-8") as f: f.write(ref_text)
    return {"status": "success", "text": ref_text}

@app.post("/synthesize")
async def synthesize(data: str = Form(...)):
    req = json.loads(data)
    gen_text = req.get("text", "")
    voice_id = req.get("voice_id", "default")
    
    p_audio = os.path.join(PROFILES_DIR, f"{voice_id}.wav")
    p_text = os.path.join(PROFILES_DIR, f"{voice_id}.txt")

    if os.path.exists(p_audio) and os.path.exists(p_text):
        ref_audio, _ = librosa.load(p_audio, sr=24000, mono=True)
        with open(p_text, "r", encoding="utf-8") as f: ref_text = f.read()
    else:
        ref_audio = np.zeros(24000, dtype=np.float32)
        ref_text = ""

    tokens = np.array([models["vocab"].get(c, 0) for c in (ref_text + gen_text)], dtype=np.int64)
    max_dur = np.array([int(len(ref_audio)/256 + 1 + len(gen_text)*2)], dtype=np.int64)
    
    pre_in = {}
    for inp in models["sess_pre"].get_inputs():
        if 'audio' in inp.name: pre_in[inp.name] = get_aligned_tensor(models["sess_pre"], 'audio', np.expand_dims(ref_audio, axis=(0,1)))
        elif 'text_ids' in inp.name: pre_in[inp.name] = get_aligned_tensor(models["sess_pre"], 'text_ids', np.expand_dims(tokens, axis=0))
        elif 'max_duration' in inp.name: pre_in[inp.name] = get_aligned_tensor(models["sess_pre"], 'max_duration', max_dur)

    pre_outs = models["sess_pre"].run(None, pre_in)
    noise = pre_outs[0]
    
    for step in range(30):
        trans_in = {}
        for i, inp in enumerate(models["sess_trans"].get_inputs()):
            if 'noise' in inp.name: trans_in[inp.name] = noise
            elif 'time_step' in inp.name: trans_in[inp.name] = get_aligned_tensor(models["sess_trans"], 'time_step', np.array([step]))
            else: trans_in[inp.name] = get_aligned_tensor(models["sess_trans"], inp.name, pre_outs[i+1])
        noise = models["sess_trans"].run(None, trans_in)[0]
        
    audio_out = models["sess_dec"].run(None, {"denoised": noise, "ref_signal_len": pre_outs[7]})[0]
    
    # --- CONVERSIÓN DE AUDIO LIMPIA (FIX RUIDO) ---
    audio_flat = audio_out.flatten()
    # 1. Normalización y Clipping
    audio_norm = np.clip(audio_flat, -1.0, 1.0)
    # 2. Conversión a Int16 (Little Endian)
    pcm_ints = (audio_norm * 32767).astype(np.int16)
    # 3. Exportar bytes crudos
    return Response(content=pcm_ints.tobytes(), media_type="audio/pcm")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)