package com.example.aittsapp.ai

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
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
            LogManager.log("Motor IA iniciado. Rangos de tensores corregidos.")
        } catch (e: Exception) { LogManager.log("ERROR INIT: " + e.message) }
    }

    override fun synthesize(text: String, profile: VoiceProfile): ByteArray? {
        if (!isReady) return null
        val env = ortEnv ?: return null
        try {
            val tokens = (profile.referenceText + text).map { it.toString() }.map { vocabMap[it] ?: 0L }.toLongArray()
            val audioData = loadRawAudio(profile.referenceAudio)
            val maxDuration = (audioData.size / 512 + 1 + text.length * 2).toLong()

            // 1. Preparar Tensores con RANGO CORRECTO (Got 2 Expected 1 Fix)
            val audioTensor = createAutoTensor(env, "audio", sessionPre!!, audioData)
            val tokenTensor = createAutoTensor(env, "text_ids", sessionPre!!, tokens)
            val durationTensor = createAutoTensor(env, "max_duration", sessionPre!!, maxDuration)

            val preInputs = mutableMapOf<String, OnnxTensor>()
            audioTensor?.let { preInputs["audio"] = it }
            tokenTensor?.let { preInputs["text_ids"] = it }
            durationTensor?.let { preInputs["max_duration"] = it }

            val preResults = sessionPre?.run(preInputs) ?: return null
            var noise = preResults.get(0) as OnnxTensor
            
            // Capturar otros 7 tensores necesarios para el transformer
            val transInputsBase = mutableMapOf<String, OnnxTensor>()
            transInputsBase["rope_cos_q"] = preResults.get(1) as OnnxTensor
            transInputsBase["rope_sin_q"] = preResults.get(2) as OnnxTensor
            transInputsBase["rope_cos_k"] = preResults.get(3) as OnnxTensor
            transInputsBase["rope_sin_k"] = preResults.get(4) as OnnxTensor
            transInputsBase["cat_mel_text"] = preResults.get(5) as OnnxTensor
            transInputsBase["cat_mel_text_drop"] = preResults.get(6) as OnnxTensor
            val refSignalLen = preResults.get(7) as OnnxTensor

            // 2. Bucle de Difusión (16 pasos)
            var currentTimeStep = 0L
            for (step in 0 until 16) {
                val timeTensor = createAutoTensor(env, "time_step", sessionTrans!!, currentTimeStep)
                val transInputs = transInputsBase.toMutableMap()
                transInputs["noise"] = noise
                timeTensor?.let { transInputs["time_step"] = it }
                
                val transResults = sessionTrans?.run(transInputs) ?: break
                val nextNoise = transResults.get(0) as OnnxTensor
                if (step > 0) noise.close()
                timeTensor?.close()
                noise = nextNoise
                currentTimeStep++
            }

            // 3. Decodificación
            val decResults = sessionDec?.run(mapOf("denoised" to noise, "ref_signal_len" to refSignalLen)) ?: return null
            return convertToPcm(decResults.get(0).value)
        } catch (e: Exception) {
            LogManager.log("ERROR IA: " + e.message)
            return null
        }
    }

    private fun createAutoTensor(env: OrtEnvironment, inputName: String, session: OrtSession, data: Any): OnnxTensor? {
        val info = session.inputInfo[inputName]?.info as? TensorInfo ?: return null
        val expectedType = info.type
        val expectedRank = info.shape.size

        return when (expectedType) {
            OnnxJavaType.FLOAT -> {
                val floats = if (data is FloatArray) data else bytesToFloats(data as? ByteArray ?: ByteArray(0))
                val shape = if (expectedRank == 3) longArrayOf(1, 1, floats.size.toLong()) else longArrayOf(floats.size.toLong())
                OnnxTensor.createTensor(env, FloatBuffer.wrap(floats), shape)
            }
            OnnxJavaType.INT64 -> {
                val longs = when (data) {
                    is LongArray -> data
                    is Long -> longArrayOf(data)
                    else -> longArrayOf(0L)
                }
                // FIX: Si el modelo espera rango 1, enviar [N]. Si espera rango 2, enviar [1, N].
                val shape = if (expectedRank == 1) longArrayOf(longs.size.toLong()) else longArrayOf(1, longs.size.toLong())
                OnnxTensor.createTensor(env, LongBuffer.wrap(longs), shape)
            }
            OnnxJavaType.INT32 -> {
                val ints = when (data) {
                    is IntArray -> data
                    is Int -> intArrayOf(data)
                    is Long -> intArrayOf(data.toInt())
                    is LongArray -> data.map { it.toInt() }.toIntArray()
                    else -> intArrayOf(0)
                }
                val shape = if (expectedRank == 1) longArrayOf(ints.size.toLong()) else longArrayOf(1, ints.size.toLong())
                OnnxTensor.createTensor(env, IntBuffer.wrap(ints), shape)
            }
            else -> null
        }
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
            context.assets.open(assetName).use { input -> FileOutputStream(file).use { output -> input.copyTo(output) } }
        }
        return file
    }

    override fun release() {
        sessionPre?.close(); sessionTrans?.close(); sessionDec?.close(); ortEnv?.close()
    }
}