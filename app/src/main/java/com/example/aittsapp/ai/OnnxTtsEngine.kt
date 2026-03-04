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
    private var context: Context? = null
    private var ortEnv: OrtEnvironment? = null
    private var sessionPre: OrtSession? = null
    private var sessionTrans: OrtSession? = null
    private var sessionDec: OrtSession? = null
    private val vocabMap = mutableMapOf<String, Long>()
    private var isReady = false

    override fun initialize(context: Context) {
        this.context = context
        LogManager.log("Inicializando motor IA...")
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            
            val vocabFile = getFileFromAssets(context, "models/base/vocab.txt")
            vocabFile.bufferedReader().readLines().forEachIndexed { index, s -> 
                if (s.isNotEmpty()) vocabMap[s] = index.toLong() 
            }

            val options = OrtSession.SessionOptions().apply {
                addConfigEntry("session.load_model_format", "ONNX")
                // OPTIMIZACIONES CRÍTICAS PARA EVITAR CUELGUES (OOM)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
                setIntraOpNumThreads(2) // Límita el estrés del procesador (Evita congelamientos)
                setInterOpNumThreads(1)
            }
            
            sessionPre = ortEnv?.createSession(getFileFromAssets(context, "models/base/F5_Preprocess.onnx").absolutePath, options)
            sessionTrans = ortEnv?.createSession(getFileFromAssets(context, "models/base/F5_Transformer.onnx").absolutePath, options)
            sessionDec = ortEnv?.createSession(getFileFromAssets(context, "models/base/F5_Decode.onnx").absolutePath, options)
            
            isReady = (sessionPre != null && sessionTrans != null && sessionDec != null)
            System.gc() // Forzar limpieza tras carga gigante
            LogManager.log("IA Lista. Memoria blindada (2 Hilos).")
        } catch (e: Exception) { 
            LogManager.log("ERROR INIT: " + e.message)
        }
    }

    override fun synthesize(text: String, profile: VoiceProfile): ByteArray? {
        val env = ortEnv ?: return null
        val ctx = context ?: return null
        if (!isReady) return null

        val prefs = ctx.getSharedPreferences("TTS_PREFS", Context.MODE_PRIVATE)
        val steps = if (prefs.getBoolean("HIGH_QUALITY", false)) 32 else 16
        
        try {
            LogManager.log("Sintetizando...")
            
            val tokens = (profile.referenceText + text).map { it.toString() }
                .map { vocabMap[it] ?: 0L }.toLongArray()
            
            val audioData = loadRawAudio(profile.referenceAudio)
            val maxDuration = (audioData.size / 512 + 1 + text.length * 2).toLong()

            // ETAPA A: Pre-procesamiento
            val preInputs = mutableMapOf<String, OnnxTensor>()
            findAndAddTensor(env, sessionPre!!, preInputs, "audio", audioData)
            findAndAddTensor(env, sessionPre!!, preInputs, "text_ids", tokens)
            findAndAddTensor(env, sessionPre!!, preInputs, "max_duration", maxDuration)

            val preResults = sessionPre?.run(preInputs) ?: return null
            var noise = preResults.get(0) as OnnxTensor
            
            // Capturar tensores de soporte para el Transformer
            val transInputsBase = mutableMapOf<String, OnnxTensor>()
            val supportKeys = listOf("rope_cos_q", "rope_sin_q", "rope_cos_k", "rope_sin_k", "cat_mel_text", "cat_mel_text_drop")
            supportKeys.forEachIndexed { i, key ->
                transInputsBase[key] = preResults.get(i + 1) as OnnxTensor
            }
            val refSignalLen = preResults.get(7) as OnnxTensor

            // ETAPA B: Bucle de Difusión Dinámico
            var currentTimeStep = 0L
            val transInputNames = sessionTrans!!.inputNames.toList()
            val noiseKey = transInputNames.find { it.contains("noise") } ?: "noise"
            val timeKey = transInputNames.find { it.contains("time_step") } ?: "time_step"

            for (step in 0 until steps) {
                val transInputs = transInputsBase.toMutableMap()
                transInputs[noiseKey] = noise
                
                val timeTensor = createAutoTensor(env, timeKey, sessionTrans!!, currentTimeStep)
                timeTensor?.let { transInputs[timeKey] = it }
                
                val transResults = sessionTrans?.run(transInputs) ?: break
                val nextNoise = transResults.get(0) as OnnxTensor
                
                if (step > 0) noise.close()
                timeTensor?.close()
                noise = nextNoise
                currentTimeStep++
            }

            // ETAPA C: Decodificación
            val decInputs = mutableMapOf<String, OnnxTensor>()
            decInputs["denoised"] = noise
            decInputs["ref_signal_len"] = refSignalLen
            
            val decResults = sessionDec?.run(decInputs) ?: return null
            return convertToPcm(decResults.get(0).value)
        } catch (e: Exception) {
            LogManager.log("ERROR IA: " + e.message)
            return null
        }
    }

    private fun findAndAddTensor(env: OrtEnvironment, session: OrtSession, map: MutableMap<String, OnnxTensor>, partialName: String, data: Any) {
        val actualName = session.inputNames.find { it.contains(partialName) } ?: partialName
        val tensor = createAutoTensor(env, actualName, session, data)
        tensor?.let { map[actualName] = it }
    }

    private fun createAutoTensor(env: OrtEnvironment, inputName: String, session: OrtSession, data: Any): OnnxTensor? {
        val info = session.inputInfo[inputName]?.info as? TensorInfo ?: return null
        val expectedType = info.type
        val expectedRank = info.shape.size

        return when (expectedType) {
            OnnxJavaType.FLOAT -> {
                val floats = bytesToFloats(data as? ByteArray ?: ByteArray(0))
                val shape = if (expectedRank == 3) longArrayOf(1, 1, floats.size.toLong()) else longArrayOf(floats.size.toLong())
                OnnxTensor.createTensor(env, FloatBuffer.wrap(floats), shape)
            }
            OnnxJavaType.INT16 -> {
                val shorts = bytesToShorts(data as? ByteArray ?: ByteArray(0))
                val shape = if (expectedRank == 3) longArrayOf(1, 1, shorts.size.toLong()) else longArrayOf(shorts.size.toLong())
                OnnxTensor.createTensor(env, ShortBuffer.wrap(shorts), shape)
            }
            OnnxJavaType.INT64 -> {
                val longs = if (data is LongArray) data else longArrayOf((data as? Long) ?: (data as? Int)?.toLong() ?: 0L)
                val shape = if (expectedRank == 1) longArrayOf(longs.size.toLong()) else longArrayOf(1, longs.size.toLong())
                OnnxTensor.createTensor(env, LongBuffer.wrap(longs), shape)
            }
            OnnxJavaType.INT32 -> {
                val ints = if (data is IntArray) data else intArrayOf((data as? Int) ?: (data as? Long)?.toInt() ?: 0)
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