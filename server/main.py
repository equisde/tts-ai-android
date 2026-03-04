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
    print("=== Iniciando Servidor TTS Profesional (Fidelity Master Fix) ===")
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
            print("Modelos cargados.")
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
            
            e_shape = inp.shape
            while len(data.shape) < len(e_shape): data = np.expand_dims(data, axis=0)
            
            if len(e_shape) == len(data.shape):
                for i in range(len(e_shape)):
                    if isinstance(e_shape[i], int) and e_shape[i] > 0 and e_shape[i] != data.shape[i]:
                        for j in range(len(data.shape)):
                            if data.shape[j] == e_shape[i]:
                                return np.transpose(data, [j if x==i else (i if x==j else x) for x in range(len(data.shape))])
            return data
    return data

@app.post("/synthesize")
async def synthesize(data: str = Form(...)):
    req = json.loads(data)
    gen_text = req.get("text", "")
    voice_id = req.get("voice_id", "default")
    
    p_audio = os.path.join(PROFILES_DIR, f"{voice_id}.wav")
    p_text = os.path.join(PROFILES_DIR, f"{voice_id}.txt")

    if not os.path.exists(p_audio):
        ref_audio = np.zeros(24000 * 2, dtype=np.float32)
        ref_text = ""
    else:
        ref_audio, _ = librosa.load(p_audio, sr=24000, mono=True)
        with open(p_text, "r", encoding="utf-8") as f: ref_text = f.read()

    full_text = (ref_text + " " + gen_text).strip()
    tokens = np.array([models["vocab"].get(c, 0) for c in full_text], dtype=np.int64)
    max_dur = np.array([int(len(ref_audio)/256 + 1 + len(gen_text)*2.5)], dtype=np.int64)
    
    # --- ETAPA A: PREPROCESS (Generación de Ruido Controlado) ---
    pre_in = {}
    for inp in models["sess_pre"].get_inputs():
        if 'audio' in inp.name: pre_in[inp.name] = align_and_cast(models["sess_pre"], 'audio', np.expand_dims(ref_audio, axis=(0,1)))
        elif 'text_ids' in inp.name: pre_in[inp.name] = align_and_cast(models["sess_pre"], 'text_ids', np.expand_dims(tokens, axis=0))
        elif 'max_duration' in inp.name: pre_in[inp.name] = align_and_cast(models["sess_pre"], 'max_duration', max_dur)

    pre_outs = models["sess_pre"].run(None, pre_in)
    pre_data = {out.name: val for out, val in zip(models["sess_pre"].get_outputs(), pre_outs)}
    
    # 1. El ruido inicial debe ser exactamente el que genera el pre-procesador
    noise = pre_outs[0]
    ref_len = pre_outs[7]
    
    # --- ETAPA B: TRANSFORMER (Difusión ODE Solver corregida) ---
    steps = 30
    dt = 1.0 / steps
    
    for step in range(steps):
        t_normalized = float(step) / steps
        
        trans_in = {}
        for inp in models["sess_trans"].get_inputs():
            if 'noise' in inp.name:
                trans_in[inp.name] = noise
            elif 'time_step' in inp.name:
                # FIX: Algunos modelos esperan el step como Float32 (0.0-1.0) 
                # y otros como Int64 (0-steps). Enviamos el tipo detectado.
                val = np.array([t_normalized if 'float' in inp.type else step], dtype=np.float32 if 'float' in inp.type else np.int64)
                trans_in[inp.name] = align_and_cast(models["sess_trans"], 'time_step', val)
            else:
                # Mapeo por nombre para soportes (Rope/Mel)
                match = next((v for k, v in pre_data.items() if any(sub in k.lower() for sub in inp.name.lower().split('_'))), None)
                if match is not None: trans_in[inp.name] = align_and_cast(models["sess_trans"], inp.name, match)
        
        # OBTENEMOS EL VECTOR DE VELOCIDAD
        v_flow = models["sess_trans"].run(None, trans_in)[0]
        
        # INTEGRACIÓN: noise_next = noise_current + (velocity * dt)
        noise = noise + (v_flow * dt)
        
    # --- ETAPA C: DECODE (Vocos con Normalización de Rango) ---
    audio_out = models["sess_dec"].run(None, {"denoised": noise, "ref_signal_len": ref_len})[0]
    
    # 1. Quitar el canal de lote [1, T] -> [T]
    audio_flat = audio_out.flatten().astype(np.float32)
    
    # 2. Normalización de Volumen (Evitar estática por amplitud alta)
    # Centramos la onda
    audio_flat -= np.mean(audio_flat)
    # Escalamos al rango audile
    if np.max(np.abs(audio_flat)) > 0:
        audio_flat = audio_flat / np.max(np.abs(audio_flat)) * 0.8
    
    return Response(content=(audio_flat * 32767).astype(np.int16).tobytes(), media_type="audio/pcm")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)