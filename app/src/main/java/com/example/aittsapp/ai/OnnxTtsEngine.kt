package com.example.aittsapp.ai

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.nio.LongBuffer

class OnnxTtsEngine : TtsEngine {
    private val TAG = "OnnxTtsEngine"
    private var ortEnv: OrtEnvironment? = null
    private var sessionPre: OrtSession? = null
    private var sessionTrans: OrtSession? = null
    private var sessionDec: OrtSession? = null
    private val vocabMap = mutableMapOf<String, Long>()
    private val SAMPLE_RATE = 24000
    private val HOP_LENGTH = 256

    override fun initialize(context: Context) {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            
            // 1. Cargar Vocabulario Real (Map String -> Long)
            val vocabFile = getFileFromAssets(context, "models/base/vocab.txt")
            vocabFile.bufferedReader().readLines().forEachIndexed { index, s -> 
                if (s.isNotEmpty()) vocabMap[s] = index.toLong() 
            }

            val options = OrtSession.SessionOptions().apply {
                addConfigEntry("session.load_model_format", "ONNX")
            }
            
            // 2. Cargar sesiones (Desde archivo para mapeo de memoria)
            sessionPre = ortEnv?.createSession(getFileFromAssets(context, "models/base/F5_Preprocess.onnx").absolutePath, options)
            sessionTrans = ortEnv?.createSession(getFileFromAssets(context, "models/base/F5_Transformer.onnx").absolutePath, options)
            sessionDec = ortEnv?.createSession(getFileFromAssets(context, "models/base/F5_Decode.onnx").absolutePath, options)
            
            Log.i(TAG, "IA inicializada con éxito. Vocabulario: \${vocabMap.size} entradas.")
        } catch (e: Exception) { 
            Log.e(TAG, "Error crítico en init: \${e.message}") 
            e.printStackTrace()
        }
    }

    override fun synthesize(text: String, profile: VoiceProfile): ByteArray? {
        val env = ortEnv ?: return null
        try {
            // 1. Tokenización (LongArray)
            val tokens = (profile.referenceText + text).map { it.toString() }
                .map { vocabMap[it] ?: 0L }.toLongArray()
            
            // 2. Audio de Referencia a Float32 (Requerido por F5-TTS)
            val refAudioFloats = loadAudioAsFloats(profile.referenceAudio)
            
            val refTextLen = profile.referenceText.length.coerceAtLeast(1)
            val refAudioLen = refAudioFloats.size / HOP_LENGTH + 1
            val maxDuration = (refAudioLen + (refAudioLen.toFloat() / refTextLen * text.length)).toLong()

            // ETAPA A: Pre-procesamiento
            val preInputs = mapOf(
                "audio" to OnnxTensor.createTensor(env, FloatBuffer.wrap(refAudioFloats), longArrayOf(1, 1, refAudioFloats.size.toLong())),
                "text_ids" to OnnxTensor.createTensor(env, LongBuffer.wrap(tokens), longArrayOf(1, tokens.size.toLong())),
                "max_duration" to OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(maxDuration)), longArrayOf(1))
            )

            val preResults = sessionPre?.run(preInputs) ?: return null
            var noise = preResults.get(0) as OnnxTensor
            val ropeCosQ = preResults.get(1) as OnnxTensor
            val ropeSinQ = preResults.get(2) as OnnxTensor
            val ropeCosK = preResults.get(3) as OnnxTensor
            val ropeSinK = preResults.get(4) as OnnxTensor
            val catMelText = preResults.get(5) as OnnxTensor
            val catMelTextDrop = preResults.get(6) as OnnxTensor
            val refSignalLen = preResults.get(7) as OnnxTensor

            // ETAPA B: Bucle de Difusión (32 pasos)
            var timeStep = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(0)), longArrayOf(1))
            for (step in 0 until 32) {
                val transResults = sessionTrans?.run(mapOf(
                    "noise" to noise, "rope_cos_q" to ropeCosQ, "rope_sin_q" to ropeSinQ,
                    "rope_cos_k" to ropeCosK, "rope_sin_k" to ropeSinK,
                    "cat_mel_text" to catMelText, "cat_mel_text_drop" to catMelTextDrop, "time_step" to timeStep
                )) ?: break
                
                // Actualizar tensores para el siguiente paso
                val nextNoise = transResults.get(0) as OnnxTensor
                val nextTime = transResults.get(1) as OnnxTensor
                
                // Liberar memoria del paso anterior
                if (step > 0) { noise.close(); timeStep.close() }
                noise = nextNoise
                timeStep = nextTime
            }

            // ETAPA C: Decodificación
            val decResults = sessionDec?.run(mapOf("denoised" to noise, "ref_signal_len" to refSignalLen)) ?: return null
            val audioOutput = decResults.get(0).value

            return convertToPcm(audioOutput)
        } catch (e: Exception) {
            Log.e(TAG, "Fallo en síntesis: \${e.message}")
            e.printStackTrace()
            return null
        }
    }

    private fun loadAudioAsFloats(file: File?): FloatArray {
        // Convierte audio a Float32 entre -1 y 1
        return if (file != null && file.exists()) {
            val bytes = file.readBytes()
            FloatArray(bytes.size / 2) { i ->
                val sample = ((bytes[i * 2 + 1].toInt() shl 8) or (bytes[i * 2].toInt() and 0xFF)).toShort()
                sample.toFloat() / 32768.0f
            }
        } else FloatArray(SAMPLE_RATE) // Silencio
    }

    private fun convertToPcm(output: Any?): ByteArray? {
        val floats = when (output) {
            is FloatArray -> output
            is Array<*> -> if (output[0] is FloatArray) output[0] as FloatArray else null
            else -> null
        } ?: return null

        val bytes = ByteArray(floats.size * 2)
        for (i in floats.indices) {
            val sample = (floats[i] * 32767).toInt().coerceIn(-32768, 32767)
            bytes[i * 2] = (sample and 0xFF).toByte()
            bytes[i * 2 + 1] = ((sample shr 8) and 0xFF).toByte()
        }
        return bytes
    }

    private fun getFileFromAssets(context: Context, assetName: String): File {
        val file = File(context.filesDir, assetName.split("/").last())
        if (!file.exists()) {
            context.assets.open(assetName).use { input ->
                FileOutputStream(file).use { output -> input.copyTo(output) }
            }
        }
        return file
    }

    override fun release() {
        sessionPre?.close(); sessionTrans?.close(); sessionDec?.close(); ortEnv?.close()
    }
}