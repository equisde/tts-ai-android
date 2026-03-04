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
            if (isReady) inspectModels()
            LogManager.log("IA Lista. Auto-detección de tipos activada.")
        } catch (e: Exception) { LogManager.log("ERROR INIT: " + e.message) }
    }

    private fun inspectModels() {
        // Log para saber qué esperan los modelos
        sessionPre?.inputInfo?.forEach { (name, info) ->
            val type = (info.info as? TensorInfo)?.type
            LogManager.log("Model Input [\$name]: \$type")
        }
    }

    override fun synthesize(text: String, profile: VoiceProfile): ByteArray? {
        if (!isReady) return null
        val env = ortEnv ?: return null
        try {
            val tokens = (profile.referenceText + text).map { it.toString() }.map { vocabMap[it] ?: 0L }.toLongArray()
            val audioData = loadRawAudio(profile.referenceAudio)
            val maxDuration = (audioData.size / 512 + 1 + text.length * 2).toLong()

            // 1. Preparar Tensores para Etapa A (Preprocess) con auto-conversión
            val audioTensor = createAutoTensor(env, "audio", sessionPre!!, audioData)
            val tokenTensor = createAutoTensor(env, "text_ids", sessionPre!!, tokens)
            val durationTensor = createAutoTensor(env, "max_duration", sessionPre!!, longArrayOf(maxDuration))

            val preInputs = mutableMapOf<String, OnnxTensor>()
            audioTensor?.let { preInputs["audio"] = it }
            tokenTensor?.let { preInputs["text_ids"] = it }
            durationTensor?.let { preInputs["max_duration"] = it }

            val preResults = sessionPre?.run(preInputs) ?: return null
            var noise = preResults.get(0) as OnnxTensor
            
            // 2. Bucle de Difusión (16 pasos)
            var timeStepVal = 0L
            for (step in 0 until 16) {
                val timeTensor = createAutoTensor(env, "time_step", sessionTrans!!, longArrayOf(timeStepVal))
                val transInputs = mutableMapOf("noise" to noise)
                timeTensor?.let { transInputs["time_step"] = it }
                
                val transResults = sessionTrans?.run(transInputs) ?: break
                val nextNoise = transResults.get(0) as OnnxTensor
                if (step > 0) noise.close()
                timeTensor?.close()
                noise = nextNoise
                timeStepVal++
            }

            // 3. Decodificación
            val decResults = sessionDec?.run(mapOf("denoised" to noise)) ?: return null
            return convertToPcm(decResults.get(0).value)
        } catch (e: Exception) {
            LogManager.log("ERROR IA: " + e.message)
            return null
        }
    }

    private fun createAutoTensor(env: OrtEnvironment, inputName: String, session: OrtSession, data: Any): OnnxTensor? {
        val info = session.inputInfo[inputName]?.info as? TensorInfo ?: return null
        val expectedType = info.type
        val shape = info.shape

        return when (expectedType) {
            OnnxJavaType.FLOAT -> {
                val floats = if (data is FloatArray) data else bytesToFloats(data as? ByteArray ?: ByteArray(0))
                OnnxTensor.createTensor(env, FloatBuffer.wrap(floats), longArrayOf(1, 1, floats.size.toLong()))
            }
            OnnxJavaType.INT16 -> {
                val shorts = if (data is ShortArray) data else bytesToShorts(data as? ByteArray ?: ByteArray(0))
                OnnxTensor.createTensor(env, ShortBuffer.wrap(shorts), longArrayOf(1, 1, shorts.size.toLong()))
            }
            OnnxJavaType.INT64 -> {
                val longs = if (data is LongArray) data else longArrayOf((data as? Int)?.toLong() ?: 0L)
                OnnxTensor.createTensor(env, LongBuffer.wrap(longs), longArrayOf(1, longs.size.toLong()))
            }
            OnnxJavaType.INT32 -> {
                val ints = if (data is LongArray) data.map { it.toInt() }.toIntArray() else intArrayOf((data as? Int) ?: 0)
                OnnxTensor.createTensor(env, IntBuffer.wrap(ints), longArrayOf(1, ints.size.toLong()))
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