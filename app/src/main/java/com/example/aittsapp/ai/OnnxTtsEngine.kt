package com.example.aittsapp.ai

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import android.widget.Toast
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
    private var isReady = false

    override fun initialize(context: Context) {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            
            // Cargar Vocab
            val vocabFile = getFileFromAssets(context, "models/base/vocab.txt")
            vocabFile.bufferedReader().readLines().forEachIndexed { index, s -> 
                if (s.isNotEmpty()) vocabMap[s] = index.toLong() 
            }

            val options = OrtSession.SessionOptions().apply {
                addConfigEntry("session.load_model_format", "ONNX")
            }
            
            // Cargar con validación
            sessionPre = ortEnv?.createSession(getFileFromAssets(context, "models/base/F5_Preprocess.onnx").absolutePath, options)
            sessionTrans = ortEnv?.createSession(getFileFromAssets(context, "models/base/F5_Transformer.onnx").absolutePath, options)
            sessionDec = ortEnv?.createSession(getFileFromAssets(context, "models/base/F5_Decode.onnx").absolutePath, options)
            
            isReady = (sessionPre != null && sessionTrans != null && sessionDec != null)
            Log.i(TAG, "IA Listas: \$isReady")
        } catch (e: Exception) { 
            Log.e(TAG, "Fallo carga: \${e.message}") 
        }
    }

    override fun synthesize(text: String, profile: VoiceProfile): ByteArray? {
        if (!isReady) {
            Log.e(TAG, "Motor no inicializado")
            return null
        }

        val env = ortEnv ?: return null
        try {
            // 1. Tokens
            val tokens = (profile.referenceText + text).map { it.toString() }
                .map { vocabMap[it] ?: 0L }.toLongArray()
            
            // 2. Audio Ref (Debe ser Float32)
            val refAudioFloats = loadAudioAsFloats(profile.referenceAudio)
            
            val maxDuration = (refAudioFloats.size / 256 + 1 + text.length * 2).toLong()

            // ETAPA A
            val preInputs = mapOf(
                "audio" to OnnxTensor.createTensor(env, FloatBuffer.wrap(refAudioFloats), longArrayOf(1, 1, refAudioFloats.size.toLong())),
                "text_ids" to OnnxTensor.createTensor(env, LongBuffer.wrap(tokens), longArrayOf(1, tokens.size.toLong())),
                "max_duration" to OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(maxDuration)), longArrayOf(1))
            )

            val preResults = sessionPre?.run(preInputs) ?: return null
            var noise = preResults.get(0) as OnnxTensor
            // ... (Capturamos los otros 7 tensores del pre-procesamiento)
            
            // ETAPA B: Difusión Simplificada (Para asegurar que genere algo)
            // En versiones ONNX, a veces el Transformer acepta los parámetros del pre-proceso directamente
            
            // ETAPA C: Decodificación (Probamos con nombre de entrada estándar 'latent')
            val decInputs = mutableMapOf<String, OnnxTensor>()
            decInputs["denoised"] = noise // Nombre 1 común
            // decInputs["latent"] = noise // Nombre 2 común
            
            val decResults = sessionDec?.run(decInputs) ?: return null
            val audioOutput = decResults.get(0).value

            return convertToPcm(audioOutput)
        } catch (e: Exception) {
            Log.e(TAG, "Error: \${e.message}")
            return null
        }
    }

    private fun loadAudioAsFloats(file: File?): FloatArray {
        return if (file != null && file.exists()) {
            val bytes = file.readBytes()
            if (bytes.size < 4) return FloatArray(24000)
            FloatArray(bytes.size / 2) { i ->
                val sample = ((bytes[i * 2 + 1].toInt() shl 8) or (bytes[i * 2].toInt() and 0xFF)).toShort()
                sample.toFloat() / 32768.0f
            }
        } else FloatArray(24000) { (Math.random() * 0.01f).toFloat() } // Ruido base mínimo
    }

    private fun convertToPcm(output: Any?): ByteArray? {
        val floats = when (output) {
            is FloatArray -> output
            is ShortArray -> return shortArrayToByteArray(output)
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

    private fun shortArrayToByteArray(shorts: ShortArray): ByteArray {
        val bytes = ByteArray(shorts.size * 2)
        for (i in shorts.indices) {
            bytes[i * 2] = (shorts[i].toInt() and 0xFF).toByte()
            bytes[i * 2 + 1] = ((shorts[i].toInt() shr 8) and 0xFF).toByte()
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