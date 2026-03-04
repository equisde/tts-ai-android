import os
import io
import time
import json
import numpy as np
import onnxruntime as ort
import soundfile as sf
import librosa
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response

app = FastAPI(title="F5-TTS AI Server (Chilean Edition)")

MODEL_DIR = "models/base"
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.txt")

vocab_map = {}
sess_pre = None
sess_trans = None
sess_dec = None

# Extraer el máximo poder de la CPU (90%) para el servidor
import multiprocessing
cpu_count = multiprocessing.cpu_count()
num_threads = max(1, int(cpu_count * 0.9))

@app.on_event("startup")
def load_models():
    global vocab_map, sess_pre, sess_trans, sess_dec
    if not os.path.exists(VOCAB_PATH):
        print(f"ATENCIÓN: Falta el modelo en {MODEL_DIR}")
        print("Debes copiar los archivos .onnx y vocab.txt a esta carpeta.")
        return
        
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            char = line.strip()
            if char:
                vocab_map[char[0] if len(char) > 0 else ""] = idx
                
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    print(f"Cargando modelos ONNX con {num_threads} hilos de CPU (90% poder)...")
    try:
        sess_pre = ort.InferenceSession(os.path.join(MODEL_DIR, "F5_Preprocess.onnx"), opts)
        sess_trans = ort.InferenceSession(os.path.join(MODEL_DIR, "F5_Transformer.onnx"), opts)
        sess_dec = ort.InferenceSession(os.path.join(MODEL_DIR, "F5_Decode.onnx"), opts)
        print("¡Modelos IA cargados y listos en el servidor!")
    except Exception as e:
        print(f"Error al cargar modelos: {e}")

@app.post("/synthesize")
async def synthesize(data: str = Form(...), audio: UploadFile = File(None)):
    if sess_pre is None:
        return Response(content=b"", status_code=500)

    req = json.loads(data)
    text = req.get("text", "")
    ref_text = req.get("reference_text", "")
    
    # 1. Cargar el audio clonado
    if audio:
        audio_bytes = await audio.read()
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=24000, mono=True)
            ref_audio_floats = y.astype(np.float32)
        except:
            # Si falla la decodificación, asumimos RAW PCM 16bit de Android
            arr = np.frombuffer(audio_bytes, dtype=np.int16)
            ref_audio_floats = arr.astype(np.float32) / 32768.0
    else:
        # Silencio
        ref_audio_floats = np.zeros(24000, dtype=np.float32)
        
    combined_text = ref_text + text
    tokens = [vocab_map.get(c, 0) for c in combined_text]
    tokens = np.array(tokens, dtype=np.int64)
    
    ref_text_len = max(1, len(ref_text))
    ref_audio_len = len(ref_audio_floats) // 256 + 1
    max_duration = int(ref_audio_len + (ref_audio_len / ref_text_len * len(text)))
    
    # --- ETAPA A: PREPROCESS ---
    pre_inputs = {
        "audio": np.expand_dims(ref_audio_floats, axis=(0,1)).astype(np.float32),
        "text_ids": np.expand_dims(tokens, axis=0).astype(np.int64),
        "max_duration": np.array([max_duration], dtype=np.int64)
    }
    
    actual_pre_inputs = {}
    for inp in sess_pre.get_inputs():
        for k, v in pre_inputs.items():
            if k in inp.name:
                if 'int32' in inp.type: v = v.astype(np.int32)
                elif 'float' in inp.type: v = v.astype(np.float32)
                elif 'int64' in inp.type: v = v.astype(np.int64)
                actual_pre_inputs[inp.name] = v

    pre_outs = sess_pre.run(None, actual_pre_inputs)
    noise = pre_outs[0]
    
    support_tensors = {}
    support_keys = ["rope_cos_q", "rope_sin_q", "rope_cos_k", "rope_sin_k", "cat_mel_text", "cat_mel_text_drop"]
    for i, key in enumerate(support_keys):
        support_tensors[key] = pre_outs[i+1]
    ref_signal_len = pre_outs[7]
    
    # --- ETAPA B: TRANSFORMER LOOP ---
    steps = 32
    time_step_val = 0
    for step in range(steps):
        trans_inputs = {k: v for k, v in support_tensors.items()}
        trans_inputs["noise"] = noise
        trans_inputs["time_step"] = np.array([time_step_val], dtype=np.int64)
        
        actual_trans_inputs = {}
        for inp in sess_trans.get_inputs():
            for k, v in trans_inputs.items():
                if k in inp.name:
                    actual_trans_inputs[inp.name] = v
                    break
                    
        trans_outs = sess_trans.run(None, actual_trans_inputs)
        noise = trans_outs[0]
        time_step_val += 1
        
    # --- ETAPA C: DECODE ---
    dec_inputs = {"denoised": noise, "ref_signal_len": ref_signal_len}
    actual_dec_inputs = {}
    for inp in sess_dec.get_inputs():
        for k, v in dec_inputs.items():
            if k in inp.name:
                actual_dec_inputs[inp.name] = v
                break
                
    dec_outs = sess_dec.run(None, actual_dec_inputs)
    audio_out = dec_outs[0]
    
    # Convertir a RAW PCM 16-bit 24kHz
    if audio_out.dtype == np.float32:
        audio_out = np.clip(audio_out, -1.0, 1.0)
        pcm = (audio_out * 32767).astype(np.int16).tobytes()
    elif audio_out.dtype == np.int16:
        pcm = audio_out.tobytes()
    else:
        pcm = audio_out.astype(np.float32).clip(-1.0, 1.0)
        pcm = (pcm * 32767).astype(np.int16).tobytes()
        
    return Response(content=pcm, media_type="audio/pcm")

if __name__ == "__main__":
    import uvicorn
    # Lanzar el servidor en todas las interfaces de red de la PC (0.0.0.0)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")