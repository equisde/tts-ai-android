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

import multiprocessing
cpu_count = multiprocessing.cpu_count()
num_threads = max(1, int(cpu_count * 0.9))

@app.on_event("startup")
def load_models():
    global vocab_map, sess_pre, sess_trans, sess_dec
    if not os.path.exists(VOCAB_PATH):
        print(f"ERROR: No se encuentra vocab.txt en {MODEL_DIR}")
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

def create_tensor(session, input_name, data):
    """Convierte datos al tipo exacto que pide el modelo ONNX."""
    for inp in session.get_inputs():
        if input_name in inp.name:
            if 'int16' in inp.type:
                # Si pide int16, escalamos floats a shorts
                return (data * 32767).astype(np.int16)
            elif 'float' in inp.type:
                return data.astype(np.float32)
            elif 'int64' in inp.type:
                return data.astype(np.int64)
            elif 'int32' in inp.type:
                return data.astype(np.int32)
    return data

@app.post("/synthesize")
async def synthesize(data: str = Form(...), audio: UploadFile = File(None)):
    req = json.loads(data)
    text = req.get("text", "")
    ref_text = req.get("reference_text", "")
    
    if audio:
        audio_bytes = await audio.read()
        y, _ = librosa.load(io.BytesIO(audio_bytes), sr=24000, mono=True)
        ref_audio = y
    else:
        ref_audio = np.zeros(24000, dtype=np.float32)
        
    tokens = np.array([vocab_map.get(c, 0) for c in (ref_text + text)], dtype=np.int64)
    max_dur = np.array([int(len(ref_audio)/256 + 1 + len(text)*2)], dtype=np.int64)
    
    # ETAPA A: Preprocess con auto-casting
    pre_inputs = {}
    for inp in sess_pre.get_inputs():
        if 'audio' in inp.name: pre_inputs[inp.name] = create_tensor(sess_pre, 'audio', np.expand_dims(ref_audio, axis=(0,1)))
        if 'text_ids' in inp.name: pre_inputs[inp.name] = create_tensor(sess_pre, 'text_ids', np.expand_dims(tokens, axis=0))
        if 'max_duration' in inp.name: pre_inputs[inp.name] = create_tensor(sess_pre, 'max_duration', max_dur)

    pre_outs = sess_pre.run(None, pre_inputs)
    noise = pre_outs[0]
    ref_len = pre_outs[7]
    
    # ETAPA B: Transformer
    for step in range(32):
        trans_inputs = {inp.name: pre_outs[i+1] for i, inp in enumerate(sess_trans.get_inputs()) if i < 6}
        trans_inputs[sess_trans.get_inputs()[6].name] = noise
        trans_inputs[sess_trans.get_inputs()[7].name] = create_tensor(sess_trans, 'time_step', np.array([step]))
        
        noise = sess_trans.run(None, trans_inputs)[0]
        
    # ETAPA C: Decode
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