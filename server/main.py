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

# Directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "base")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)

models = {"vocab": {}, "sess_pre": None, "sess_trans": None, "sess_dec": None, "whisper": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=== Iniciando Servidor TTS (Smart Mapping Edition) ===")
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
    """Ajusta tipos y dimensiones de tensores para que encajen en el modelo."""
    for inp in session.get_inputs():
        if input_name in inp.name:
            # 1. Ajuste de Tipo
            if 'int16' in inp.type: data = (data * 32767).astype(np.int16)
            elif 'float' in inp.type: data = data.astype(np.float32)
            elif 'int64' in inp.type: data = data.astype(np.int64)
            elif 'int32' in inp.type: data = data.astype(np.int32)
            
            # 2. Ajuste de RANGO (Rank / Unsqueeze)
            e_shape = inp.shape
            while len(data.shape) < len(e_shape):
                data = np.expand_dims(data, axis=0)

            # 3. Ajuste de DIMENSIONES (Transpose)
            a_shape = data.shape
            if len(e_shape) == len(a_shape):
                for i in range(len(e_shape)):
                    target = e_shape[i]
                    if isinstance(target, int) and target > 0 and target != a_shape[i]:
                        for j in range(len(a_shape)):
                            if a_shape[j] == target:
                                axes = list(range(len(a_shape)))
                                axes[i], axes[j] = axes[j], axes[i]
                                data = np.transpose(data, axes)
                                break
            return data
    return data

@app.post("/synthesize")
async def synthesize(data: str = Form(...), audio: UploadFile = File(None)):
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

    tokens = np.array([models["vocab"].get(c, 0) for c in (ref_text + gen_text)], dtype=np.int64)
    max_dur = np.array([int(len(ref_audio)/256 + 1 + len(gen_text)*2)], dtype=np.int64)
    
    # --- ETAPA A: PREPROCESS ---
    pre_in = {}
    for inp in models["sess_pre"].get_inputs():
        if 'audio' in inp.name: pre_in[inp.name] = align_and_cast(models["sess_pre"], inp.name, np.expand_dims(ref_audio, axis=(0,1)))
        elif 'text_ids' in inp.name: pre_in[inp.name] = align_and_cast(models["sess_pre"], inp.name, np.expand_dims(tokens, axis=0))
        elif 'max_duration' in inp.name: pre_in[inp.name] = align_and_cast(models["sess_pre"], inp.name, max_dur)

    pre_outs_list = models["sess_pre"].run(None, pre_in)
    # Crear diccionario nombre_salida -> valor_salida
    pre_outs = {out.name: val for out, val in zip(models["sess_pre"].get_outputs(), pre_outs_list)}
    
    # Identificar tensores clave
    noise_key = next(n for n in pre_outs.keys() if 'noise' in n or 'denoised' in n)
    ref_len_key = next(n for n in pre_outs.keys() if 'ref' in n and 'len' in n)
    
    noise = pre_outs[noise_key]
    ref_len = pre_outs[ref_len_key]
    
    # --- ETAPA B: TRANSFORMER (Difusión con Mapeo por Nombre) ---
    for step in range(30):
        trans_in = {}
        for inp in models["sess_trans"].get_inputs():
            if 'noise' in inp.name:
                trans_in[inp.name] = noise
            elif 'time_step' in inp.name:
                trans_in[inp.name] = align_and_cast(models["sess_trans"], inp.name, np.array([step], dtype=np.int64))
            else:
                # BUSCAR COINCIDENCIA DE NOMBRE (cat_mel, rope, etc.)
                # El pre-procesador devuelve nombres como 'rope_cos_q', buscamos eso en la entrada del transformer
                match_val = None
                for out_name, out_val in pre_outs.items():
                    if out_name in inp.name or inp.name in out_name:
                        match_val = out_val
                        break
                
                if match_val is not None:
                    trans_in[inp.name] = align_and_cast(models["sess_trans"], inp.name, match_val)
                else:
                    # Si no hay match por nombre, lanzamos un error descriptivo
                    print(f"ERROR: No se encontró tensor de soporte para entrada: {inp.name}")
        
        noise = models["sess_trans"].run(None, trans_in)[0]
        
    # --- ETAPA C: DECODE ---
    audio_out = models["sess_dec"].run(None, {"denoised": noise, "ref_signal_len": ref_len})[0]
    pcm = (audio_out.flatten() * 32767).astype(np.int16).tobytes()
    return Response(content=pcm, media_type="audio/pcm")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)