import os
import io
import json
import numpy as np
import onnxruntime as ort
import librosa
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from contextlib import asynccontextmanager

# Directorios y rutas
MODEL_DIR = "models/base"
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.txt")

# Variables globales para modelos
models = {
    "vocab": {},
    "sess_pre": None,
    "sess_trans": None,
    "sess_dec": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=== Iniciando Servidor IA (Chilean Edition) ===")
    if not os.path.exists(VOCAB_PATH):
        print(f"ERROR: No se encuentra {VOCAB_PATH}")
    else:
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
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
            print(f"Modelos cargados con éxito. Hilos: {num_threads}")
        except Exception as e:
            print(f"Error cargando modelos: {e}")
    yield
    print("Cerrando servidor...")

app = FastAPI(title="F5-TTS AI Server", lifespan=lifespan)

def get_aligned_tensor(session, input_name, data):
    """Detecta tipo y forma (shape) esperada por ONNX y alinea los datos."""
    for inp in session.get_inputs():
        if input_name in inp.name:
            # 1. Alineación de TIPO
            if 'int16' in inp.type:
                data = (data * 32767).astype(np.int16) if data.dtype == np.float32 else data.astype(np.int16)
            elif 'float' in inp.type:
                data = data.astype(np.float32)
            elif 'int64' in inp.type:
                data = data.astype(np.int64)
            elif 'int32' in inp.type:
                data = data.astype(np.int32)
            
            # 2. Alineación de DIMENSIONES (Shape)
            expected_shape = inp.shape
            actual_shape = data.shape
            
            if len(expected_shape) == len(actual_shape):
                # Si las dimensiones están invertidas (ej: seq vs dim)
                for i in range(len(expected_shape)):
                    if isinstance(expected_shape[i], int) and expected_shape[i] > 0:
                        if expected_shape[i] != actual_shape[i]:
                            # Intentamos encontrar dónde está la dimensión correcta
                            for j in range(i + 1, len(actual_shape)):
                                if actual_shape[j] == expected_shape[i]:
                                    print(f"DEBUG: Transponiendo {input_name} ({i}<->{j})")
                                    # Generar orden de transposición dinámico
                                    axes = list(range(len(actual_shape)))
                                    axes[i], axes[j] = axes[j], axes[i]
                                    data = np.transpose(data, axes)
                                    break
            return data
    return data

@app.post("/synthesize")
async def synthesize(data: str = Form(...), audio: UploadFile = File(None)):
    if not models["sess_pre"]: return Response(content=b"Modelos no cargados", status_code=500)
    
    try: req = json.loads(data)
    except: return Response(content=b"Error JSON", status_code=400)

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
        
    tokens = np.array([models["vocab"].get(c, 0) for c in (ref_text + text)], dtype=np.int64)
    max_dur = np.array([int(len(ref_audio)/256 + 1 + len(text)*2)], dtype=np.int64)
    
    # ETAPA A: PREPROCESS
    pre_inputs = {}
    for inp in models["sess_pre"].get_inputs():
        if 'audio' in inp.name: pre_inputs[inp.name] = get_aligned_tensor(models["sess_pre"], 'audio', np.expand_dims(ref_audio, axis=(0,1)))
        elif 'text_ids' in inp.name: pre_inputs[inp.name] = get_aligned_tensor(models["sess_pre"], 'text_ids', np.expand_dims(tokens, axis=0))
        elif 'max_duration' in inp.name: pre_inputs[inp.name] = get_aligned_tensor(models["sess_pre"], 'max_duration', max_dur)

    pre_outs = models["sess_pre"].run(None, pre_inputs)
    noise = pre_outs[0]
    ref_len = pre_outs[7]
    
    # ETAPA B: TRANSFORMER (Difusión con auto-alineación de soporte)
    try:
        for step in range(32):
            trans_inputs = {}
            for i, inp in enumerate(models["sess_trans"].get_inputs()):
                if 'noise' in inp.name: trans_inputs[inp.name] = noise
                elif 'time_step' in inp.name: trans_inputs[inp.name] = get_aligned_tensor(models["sess_trans"], 'time_step', np.array([step]))
                else:
                    # Tensores de soporte (cos/sin/mel) - i va de 0 a 5
                    val = pre_outs[i+1]
                    trans_inputs[inp.name] = get_aligned_tensor(models["sess_trans"], inp.name, val)
            
            noise = models["sess_trans"].run(None, trans_inputs)[0]
    except Exception as e:
        print(f"Error en Transformer: {e}")
        return Response(content=f"Error: {str(e)}".encode(), status_code=500)
        
    # ETAPA C: DECODE
    dec_inputs = {}
    for inp in models["sess_dec"].get_inputs():
        if 'denoised' in inp.name: dec_inputs[inp.name] = noise
        if 'ref_signal_len' in inp.name: dec_inputs[inp.name] = ref_len
        
    audio_out = models["sess_dec"].run(None, dec_inputs)[0]
    pcm = (audio_out.flatten() * 32767).astype(np.int16).tobytes()
    return Response(content=pcm, media_type="audio/pcm")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)