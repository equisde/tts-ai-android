import os
import io
import json
import numpy as np
import onnxruntime as ort
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

import multiprocessing
cpu_count = multiprocessing.cpu_count()
num_threads = max(1, int(cpu_count * 0.9))

@app.on_event("startup")
def load_models():
    global vocab_map, sess_pre, sess_trans, sess_dec
    if not os.path.exists(VOCAB_PATH):
        print(f"ERROR: Falta vocabulario.")
        return
        
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            char = line.strip()
            if char: vocab_map[char[0]] = idx
                
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    
    try:
        sess_pre = ort.InferenceSession(os.path.join(MODEL_DIR, "F5_Preprocess.onnx"), opts)
        sess_trans = ort.InferenceSession(os.path.join(MODEL_DIR, "F5_Transformer.onnx"), opts)
        sess_dec = ort.InferenceSession(os.path.join(MODEL_DIR, "F5_Decode.onnx"), opts)
        print(f"Servidor IA listo con {num_threads} hilos.")
    except Exception as e:
        print(f"Error carga: {e}")

def align_tensor(data, expected_shape, name=""):
    """Ajusta las dimensiones del tensor para que coincidan con el modelo."""
    actual_shape = data.shape
    if len(actual_shape) != len(expected_shape):
        return data # No podemos hacer mucho si el rango es distinto
        
    # FIX: Si las dimensiones están invertidas (común en RoPE tensors)
    # Ej: Espera [1, 1, 260, 64] pero recibe [1, 1, 64, 260]
    for i in range(len(actual_shape)):
        if expected_shape[i] != -1 and expected_shape[i] != actual_shape[i]:
            # Buscamos si el valor esperado está en otra posición
            if i + 1 < len(actual_shape) and actual_shape[i+1] == expected_shape[i]:
                print(f"Alineando tensor {name}: Transponiendo dimensiones.")
                # Transponemos las dos últimas dimensiones si es necesario
                return np.transpose(data, (0, 1, 3, 2))
    return data

@app.post("/synthesize")
async def synthesize(data: str = Form(...), audio: UploadFile = File(None)):
    req = json.loads(data)
    text = req.get("text", "")
    ref_text = req.get("reference_text", "")
    
    if audio:
        audio_bytes = await audio.read()
        try:
            y, _ = librosa.load(io.BytesIO(audio_bytes), sr=24000, mono=True)
            ref_audio = y
        except:
            arr = np.frombuffer(audio_bytes, dtype=np.int16)
            ref_audio = arr.astype(np.float32) / 32768.0
    else:
        ref_audio = np.zeros(24000, dtype=np.float32)
        
    tokens = np.array([vocab_map.get(c, 0) for c in (ref_text + text)], dtype=np.int64)
    max_dur = np.array([int(len(ref_audio)/256 + 1 + len(text)*2)], dtype=np.int64)
    
    # ETAPA A: PREPROCESS
    pre_inputs = {}
    for inp in sess_pre.get_inputs():
        if 'audio' in inp.name: pre_inputs[inp.name] = np.expand_dims(ref_audio, axis=(0,1)).astype(np.float32)
        if 'text_ids' in inp.name: pre_inputs[inp.name] = np.expand_dims(tokens, axis=0).astype(np.int64)
        if 'max_duration' in inp.name: pre_inputs[inp.name] = max_dur.astype(np.int64)

    pre_outs = sess_pre.run(None, pre_inputs)
    # Mapeo por nombre de salida de Preprocess a entrada de Transformer
    # 0:noise, 1:rope_cos_q, 2:rope_sin_q, 3:rope_cos_k, 4:rope_sin_k, 5:cat_mel, 6:cat_mel_drop, 7:ref_len
    
    noise = pre_outs[0]
    ref_len = pre_outs[7]
    
    # ETAPA B: TRANSFORMER (Difusión)
    for step in range(32):
        trans_inputs = {}
        # Mapeamos dinámicamente los tensores de soporte
        for i, inp in enumerate(sess_trans.get_inputs()):
            if 'noise' in inp.name: trans_inputs[inp.name] = noise
            elif 'time_step' in inp.name: trans_inputs[inp.name] = np.array([step], dtype=np.int64)
            else:
                # Tensores de soporte (cos/sin/mel)
                # i suele ir de 0 a 5 para los soportes en el orden correcto
                val = pre_outs[i+1]
                # APLICAMOS EL FIX DE DIMENSIONES AQUÍ
                trans_inputs[inp.name] = align_tensor(val, inp.shape, inp.name)
        
        noise = sess_trans.run(None, trans_inputs)[0]
        
    # ETAPA C: DECODE
    dec_inputs = {}
    for inp in sess_dec.get_inputs():
        if 'denoised' in inp.name: dec_inputs[inp.name] = noise
        if 'ref_signal_len' in inp.name: dec_inputs[inp.name] = ref_len
        
    audio_out = sess_dec.run(None, dec_inputs)[0]
    pcm = (audio_out.flatten() * 32767).astype(np.int16).tobytes()
    return Response(content=pcm, media_type="audio/pcm")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)