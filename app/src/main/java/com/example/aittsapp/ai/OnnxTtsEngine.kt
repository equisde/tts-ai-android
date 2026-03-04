package com.example.aittsapp.ai

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import com.example.aittsapp.engine.LogManager
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.nio.ShortBuffer

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
            val vocabFile = getFileFromAssets(context, "models/base/vocab.txt")
            vocabFile.bufferedReader().readLines().forEachIndexed { index, s -> if (s.isNotEmpty()) vocabMap[s] = index.toLong() }

            val options = OrtSession.SessionOptions().apply { addConfigEntry("session.load_model_format", "ONNX") }
            sessionPre = ortEnv?.createSession(getFileFromAssets(context, "models/base/F5_Preprocess.onnx").absolutePath, options)
            sessionTrans = ortEnv?.createSession(getFileFromAssets(context, "models/base/F5_Transformer.onnx").absolutePath, options)
            sessionDec = ortEnv?.createSession(getFileFromAssets(context, "models/base/F5_Decode.onnx").absolutePath, options)
            isReady = (sessionPre != null && sessionTrans != null && sessionDec != null)
            LogManager.log("IA Lista para sintetizar: " + isReady)
        } catch (e: Exception) { LogManager.log("ERROR INIT: " + e.message) }
    }

    override fun synthesize(text: String, profile: VoiceProfile): ByteArray? {
        if (!isReady) return null
        val env = ortEnv ?: return null
        try {
            val tokens = (profile.referenceText + text).map { it.toString() }.map { vocabMap[it] ?: 0L }.toLongArray()
            val audioData = loadRawAudio(profile.referenceAudio)
            val maxDuration = (audioData.size / 512 + 1 + text.length * 2).toLong()

            // REINTENTO INTELIGENTE DE TIPO DE DATO
            return trySynthesize(env, tokens, audioData, maxDuration)
        } catch (e: Exception) {
            LogManager.log("ERROR SÍNTESIS: " + e.message)
            return null
        }
    }

    private fun trySynthesize(env: OrtEnvironment, tokens: LongArray, audioData: ByteArray, maxDuration: Long): ByteArray? {
        // Intentar primero con Float32 (Estándar IA)
        try {
            LogManager.log("Intentando con Float32...")
            val floats = bytesToFloats(audioData)
            return runPipeline(env, tokens, floats, maxDuration)
        } catch (e: Exception) {
            val msg = e.message ?: ""
            if (msg.contains("int16", ignoreCase = true)) {
                LogManager.log("Detectado requerimiento Int16. Reintentando...")
                val shorts = bytesToShorts(audioData)
                return runPipeline(env, tokens, shorts, maxDuration)
            }
            LogManager.log("Fallo en reintento: " + msg)
            return null
        }
    }

    private fun runPipeline(env: OrtEnvironment, tokens: LongArray, audio: Any, maxDuration: Long): ByteArray? {
        // Generar Tensor dinámico según el tipo detectado
        val audioTensor = when (audio) {
            is FloatArray -> OnnxTensor.createTensor(env, FloatBuffer.wrap(audio), longArrayOf(1, 1, audio.size.toLong()))
            is ShortArray -> OnnxTensor.createTensor(env, ShortBuffer.wrap(audio), longArrayOf(1, 1, audio.size.toLong()))
            else -> return null
        }

        val preInputs = mapOf(
            "audio" to audioTensor,
            "text_ids" to OnnxTensor.createTensor(env, LongBuffer.wrap(tokens), longArrayOf(1, tokens.size.toLong())),
            "max_duration" to OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(maxDuration)), longArrayOf(1))
        )

        val preResults = sessionPre?.run(preInputs) ?: return null
        var noise = preResults.get(0) as OnnxTensor
        
        // Bucle de Difusión (16 pasos)
        var timeStep = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(0)), longArrayOf(1))
        for (step in 0 until 16) {
            val transResults = sessionTrans?.run(mapOf("noise" to noise, "time_step" to timeStep)) ?: break
            val nextNoise = transResults.get(0) as OnnxTensor
            val nextTime = transResults.get(1) as OnnxTensor
            if (step > 0) { noise.close(); timeStep.close() }
            noise = nextNoise
            timeStep = nextTime
        }

        val decResults = sessionDec?.run(mapOf("denoised" to noise)) ?: return null
        return convertToPcm(decResults.get(0).value)
    }

    private fun loadRawAudio(file: File?): ByteArray {
        return if (file != null && file.exists()) file.readBytes() else ByteArray(48000)
    }

    private fun bytesToFloats(bytes: ByteArray): FloatArray {
        return FloatArray(bytes.size / 2) { i ->
            val sample = ((bytes[i * 2 + 1].toInt() shl 8) or (bytes[i * 2].toInt() and 0xFF)).toShort()
            sample.toFloat() / 32768.0f
        }
    }

    private fun bytesToShorts(bytes: ByteArray): ShortArray {
        return ShortArray(bytes.size / 2) { i ->
            ((bytes[i * 2 + 1].toInt() shl 8) or (bytes[i * 2].toInt() and 0xFF)).toShort()
        }
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
            context.assets.open(assetName).use { input -> FileOutputStream(file).use { output -> input.copyTo(output) } }
        }
        return file
    }

    override fun release() {
        sessionPre?.close(); sessionTrans?.close(); sessionDec?.close(); ortEnv?.close()
    }
}