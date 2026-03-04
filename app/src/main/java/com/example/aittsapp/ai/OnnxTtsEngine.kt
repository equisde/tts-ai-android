package com.example.aittsapp.ai

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.nio.ShortBuffer

class OnnxTtsEngine : TtsEngine {
    private val TAG = "OnnxTtsEngine"
    private var ortEnv: OrtEnvironment? = null
    private var sessionPre: OrtSession? = null
    private var sessionTrans: OrtSession? = null
    private var sessionDec: OrtSession? = null
    private val vocabMap = mutableMapOf<Char, Int>()
    private val SAMPLE_RATE = 24000
    private val HOP_LENGTH = 256

    override fun initialize(context: Context) {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val vocabFile = getFileFromAssets(context, "models/base/vocab.txt")
            vocabFile.bufferedReader().readLines().forEachIndexed { index, s -> if (s.isNotEmpty()) vocabMap[s[0]] = index }

            val pathPre = getFileFromAssets(context, "models/base/F5_Preprocess.onnx").absolutePath
            val pathTrans = getFileFromAssets(context, "models/base/F5_Transformer.onnx").absolutePath
            val pathDec = getFileFromAssets(context, "models/base/F5_Decode.onnx").absolutePath

            val options = OrtSession.SessionOptions().apply { addConfigEntry("session.load_model_format", "ONNX") }
            sessionPre = ortEnv?.createSession(pathPre, options)
            sessionTrans = ortEnv?.createSession(pathTrans, options)
            sessionDec = ortEnv?.createSession(pathDec, options)
            Log.i(TAG, "IA inicializada correctamente.")
        } catch (e: Exception) { Log.e(TAG, "Error init: \${e.message}") }
    }

    override fun synthesize(text: String, profile: VoiceProfile): ByteArray? {
        val env = ortEnv ?: return null
        try {
            val combinedText = profile.referenceText + text
            val tokens = combinedText.map { vocabMap[it] ?: 0 }.toIntArray()
            
            // Cargar audio de referencia (CRÍTICO: Si no hay, la IA no genera nada)
            val refAudioShorts = if (profile.referenceAudio != null) loadAudioAsShorts(profile.referenceAudio) 
                                 else loadReferenceFromAssets(env, profile.gender)

            val refTextLen = profile.referenceText.length.coerceAtLeast(1)
            val refAudioLen = refAudioShorts.size / HOP_LENGTH + 1
            val maxDuration = (refAudioLen + (refAudioLen.toFloat() / refTextLen * text.length)).toLong()

            val preInputs = mapOf(
                "audio" to OnnxTensor.createTensor(env, ShortBuffer.wrap(refAudioShorts), longArrayOf(1, 1, refAudioShorts.size.toLong())),
                "text_ids" to OnnxTensor.createTensor(env, IntBuffer.wrap(tokens), longArrayOf(1, tokens.size.toLong())),
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

            var timeStep = OnnxTensor.createTensor(env, IntBuffer.wrap(intArrayOf(0)), longArrayOf(1))
            for (step in 0 until 32) {
                val transResults = sessionTrans?.run(mapOf(
                    "noise" to noise, "rope_cos_q" to ropeCosQ, "rope_sin_q" to ropeSinQ,
                    "rope_cos_k" to ropeCosK, "rope_sin_k" to ropeSinK,
                    "cat_mel_text" to catMelText, "cat_mel_text_drop" to catMelTextDrop, "time_step" to timeStep
                )) ?: break
                noise = transResults.get(0) as OnnxTensor
                timeStep = transResults.get(1) as OnnxTensor
            }

            val decResults = sessionDec?.run(mapOf("denoised" to noise, "ref_signal_len" to refSignalLen)) ?: return null
            val audioOutput = decResults.get(0).value

            // Manejo dinámico de tipos (FloatArray vs ShortArray)
            return when (audioOutput) {
                is FloatArray -> floatArrayToByteArray(audioOutput)
                is ShortArray -> shortArrayToByteArray(audioOutput)
                is Array<*> -> if (audioOutput[0] is FloatArray) floatArrayToByteArray(audioOutput[0] as FloatArray) else null
                else -> {
                    Log.e(TAG, "Tipo de audio desconocido: \${audioOutput?.javaClass?.simpleName}")
                    null
                }
            }
        } catch (e: Exception) { Log.e(TAG, "Error síntesis: \${e.message}"); return null }
    }

    private fun loadReferenceFromAssets(env: OrtEnvironment, gender: Gender): ShortArray {
        // En una app real aquí cargaríamos un .wav de assets. Por ahora generamos ruido rosa de 2s para que la IA tenga contexto.
        return ShortArray(SAMPLE_RATE * 2) { (Math.random() * 1000).toInt().toShort() }
    }

    private fun floatArrayToByteArray(floats: FloatArray): ByteArray {
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
            context.assets.open(assetName).use { input -> FileOutputStream(file).use { output -> input.copyTo(output) } }
        }
        return file
    }

    private fun loadAudioAsShorts(file: File?): ShortArray {
        return if (file != null && file.exists()) {
            val bytes = file.readBytes()
            if (bytes.size < 2) ShortArray(SAMPLE_RATE)
            else ShortArray(bytes.size / 2) { i -> ((bytes[i * 2 + 1].toInt() shl 8) or (bytes[i * 2].toInt() and 0xFF)).toShort() }
        } else ShortArray(SAMPLE_RATE)
    }

    override fun release() {
        sessionPre?.close(); sessionTrans?.close(); sessionDec?.close(); ortEnv?.close()
    }
}