import os
import io
import json
import numpy as np
import onnxruntime as ort
import librosa
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from contextlib import asynccontextmanager

# Directorios de producción
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "base")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)

models = {"vocab": {}, "sess_pre": None, "sess_trans": None, "sess_dec": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=== Iniciando Servidor de Identidad Vocal IA ===")
    vocab_path = os.path.join(MODEL_DIR, "vocab.txt")
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f.readlines()):
                char = line.strip()
                if char: models["vocab"][char[0]] = idx
        
        import multiprocessing
        num_threads = max(1, int(multiprocessing.cpu_count() * 0.9))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        
        try:
            models["sess_pre"] = ort.InferenceSession(os.path.join(MODEL_DIR, "F5_Preprocess.onnx"), opts)
            models["sess_trans"] = ort.InferenceSession(os.path.join(MODEL_DIR, "F5_Transformer.onnx"), opts)
            models["sess_dec"] = ort.InferenceSession(os.path.join(MODEL_DIR, "F5_Decode.onnx"), opts)
            print(f"Modelos cargados. Potencia CPU: {num_threads} hilos.")
        except Exception as e:
            print(f"Error carga: {e}")
    yield

app = FastAPI(title="F5-TTS Voice Identity Server", lifespan=lifespan)

def align_tensor(session, input_name, data):
    for inp in session.get_inputs():
        if input_name in inp.name:
            if 'int16' in inp.type:
                data = (data * 32767).astype(np.int16) if data.dtype == np.float32 else data.astype(np.int16)
            elif 'float' in inp.type: data = data.astype(np.float32)
            elif 'int64' in inp.type: data = data.astype(np.int64)
            elif 'int32' in inp.type: data = data.astype(np.int32)
            
            e_shape = inp.shape
            a_shape = data.shape
            if len(e_shape) == len(a_shape):
                for i in range(len(e_shape)):
                    if isinstance(e_shape[i], int) and e_shape[i] > 0 and e_shape[i] != a_shape[i]:
                        for j in range(len(a_shape)):
                            if a_shape[j] == e_shape[i]:
                                return np.transpose(data, list(range(len(a_shape))).apply { this[i]=j; this[j]=i }) # Simplificado
            return data
    return data

@app.post("/synthesize")
async def synthesize(data: str = Form(...), audio: UploadFile = File(None)):
    req = json.loads(data)
    gen_text = req.get("text", "")
    voice_id = req.get("voice_id", "default")
    ref_text = req.get("reference_text", "")
    
    # LÓGICA DE IDENTIDAD VOCAL
    profile_path = os.path.join(PROFILES_DIR, f"{voice_id}.wav")
    ref_text_path = os.path.join(PROFILES_DIR, f"{voice_id}.txt")

    # 1. Si recibimos audio nuevo (Clonación en caliente), lo guardamos
    if audio:
        audio_bytes = await audio.read()
        with open(profile_path, "wb") as f: f.write(audio_bytes)
        if ref_text:
            with open(ref_text_path, "w", encoding="utf-8") as f: f.write(ref_text)
    
    # 2. Cargar audio de referencia del perfil solicitado
    if os.path.exists(profile_path):
        ref_audio, _ = librosa.load(profile_path, sr=24000, mono=True)
        if not ref_text and os.path.exists(ref_text_path):
            with open(ref_text_path, "r", encoding="utf-8") as f: ref_text = f.read()
    else:
        # Fallback a silencio o voz base si no existe el perfil
        ref_audio = np.zeros(24000, dtype=np.float32)
        ref_text = ""

    tokens = np.array([models["vocab"].get(c, 0) for c in (ref_text + gen_text)], dtype=np.int64)
    max_dur = np.array([int(len(ref_audio)/256 + 1 + len(gen_text)*2)], dtype=np.int64)
    
    # Pipeline IA (A -> B -> C)
    pre_inputs = {}
    for inp in models["sess_pre"].get_inputs():
        if 'audio' in inp.name: pre_inputs[inp.name] = align_tensor(models["sess_pre"], 'audio', np.expand_dims(ref_audio, axis=(0,1)))
        if 'text_ids' in inp.name: pre_inputs[inp.name] = align_tensor(models["sess_pre"], 'text_ids', np.expand_dims(tokens, axis=0))
        if 'max_duration' in inp.name: pre_inputs[inp.name] = align_tensor(models["sess_pre"], 'max_duration', max_dur)

    pre_outs = models["sess_pre"].run(None, pre_inputs)
    noise = pre_outs[0]
    
    for step in range(30):
        trans_inputs = {inp.name: pre_outs[i+1] for i, inp in enumerate(models["sess_trans"].get_inputs()) if i < 6}
        trans_inputs[models["sess_trans"].get_inputs()[6].name] = noise
        trans_inputs[models["sess_trans"].get_inputs()[7].name] = align_tensor(models["sess_trans"], 'time_step', np.array([step]))
        noise = models["sess_trans"].run(None, trans_inputs)[0]
        
    audio_out = models["sess_dec"].run(None, {"denoised": noise, "ref_signal_len": pre_outs[7]})[0]
    return Response(content=(audio_out.flatten() * 32767).astype(np.int16).tobytes(), media_type="audio/pcm")
