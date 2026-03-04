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
import java.nio.LongBuffer

class OnnxTtsEngine : TtsEngine {
    private var ortEnv: OrtEnvironment? = null
    private var sessionPre: OrtSession? = null
    private var sessionTrans: OrtSession? = null
    private var sessionDec: OrtSession? = null
    private val vocabMap = mutableMapOf<String, Long>()
    private var isReady = false

    override fun initialize(context: Context) {
        LogManager.log("Iniciando Verificación de IA...")
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            
            val vocabFile = getFileFromAssets(context, "models/base/vocab.txt")
            val preFile = getFileFromAssets(context, "models/base/F5_Preprocess.onnx")
            val transFile = getFileFromAssets(context, "models/base/F5_Transformer.onnx")
            val decFile = getFileFromAssets(context, "models/base/F5_Decode.onnx")

            LogManager.log("Archivos listos. Cargando Vocab...")
            vocabFile.bufferedReader().readLines().forEachIndexed { index, s -> 
                if (s.isNotEmpty()) vocabMap[s] = index.toLong() 
            }

            val options = OrtSession.SessionOptions().apply {
                addConfigEntry("session.load_model_format", "ONNX")
            }
            
            LogManager.log("Cargando Preprocess (" + (preFile.length() / 1024) + " KB)...")
            sessionPre = ortEnv?.createSession(preFile.absolutePath, options)
            
            LogManager.log("Cargando Transformer (" + (transFile.length() / 1024 / 1024) + " MB)...")
            sessionTrans = ortEnv?.createSession(transFile.absolutePath, options)
            
            LogManager.log("Cargando Decode (" + (decFile.length() / 1024 / 1024) + " MB)...")
            sessionDec = ortEnv?.createSession(decFile.absolutePath, options)
            
            isReady = (sessionPre != null && sessionTrans != null && sessionDec != null)
            LogManager.log("SISTEMA LISTO: " + isReady)
        } catch (e: Exception) { 
            LogManager.log("ERROR CRÍTICO: " + e.message)
            Log.e("OnnxTtsEngine", "Init failed", e)
        }
    }

    override fun synthesize(text: String, profile: VoiceProfile): ByteArray? {
        if (!isReady) return null
        val env = ortEnv ?: return null
        try {
            LogManager.log("Sintetizando: " + text.take(20) + "...")
            
            val tokens = (profile.referenceText + text).map { it.toString() }
                .map { vocabMap[it] ?: 0L }.toLongArray()
            
            val refAudioFloats = loadAudioAsFloats(profile.referenceAudio)
            val maxDuration = (refAudioFloats.size / 256 + 1 + text.length * 2).toLong()

            val preInputs = mapOf(
                "audio" to OnnxTensor.createTensor(env, FloatBuffer.wrap(refAudioFloats), longArrayOf(1, 1, refAudioFloats.size.toLong())),
                "text_ids" to OnnxTensor.createTensor(env, LongBuffer.wrap(tokens), longArrayOf(1, tokens.size.toLong())),
                "max_duration" to OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(maxDuration)), longArrayOf(1))
            )

            val preResults = sessionPre?.run(preInputs) ?: return null
            var noise = preResults.get(0) as OnnxTensor
            
            LogManager.log("Difusión en curso...")
            var timeStep = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(0)), longArrayOf(1))
            for (step in 0 until 16) {
                val transResults = sessionTrans?.run(mapOf("noise" to noise, "time_step" to timeStep)) ?: break
                val nextNoise = transResults.get(0) as OnnxTensor
                val nextTime = transResults.get(1) as OnnxTensor
                if (step > 0) { noise.close(); timeStep.close() }
                noise = nextNoise
                timeStep = nextTime
            }

            LogManager.log("Decodificando audio...")
            val decResults = sessionDec?.run(mapOf("denoised" to noise)) ?: return null
            val audioOutput = decResults.get(0).value

            return convertToPcm(audioOutput)
        } catch (e: Exception) {
            LogManager.log("ERROR SÍNTESIS: " + e.message)
            return null
        }
    }

    private fun loadAudioAsFloats(file: File?): FloatArray {
        return if (file != null && file.exists()) {
            val bytes = file.readBytes()
            FloatArray(bytes.size / 2) { i ->
                val sample = ((bytes[i * 2 + 1].toInt() shl 8) or (bytes[i * 2].toInt() and 0xFF)).toShort()
                sample.toFloat() / 32768.0f
            }
        } else FloatArray(24000) { 0.01f }
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
        val fileName = assetName.split("/").last()
        val file = File(context.filesDir, fileName)
        
        if (!file.exists() || file.length() < 100) {
            LogManager.log("Instalando: " + fileName + "...")
            context.assets.open(assetName).use { input ->
                FileOutputStream(file).use { output ->
                    input.copyTo(output)
                }
            }
            LogManager.log("Instalación exitosa: " + fileName)
        }
        return file
    }

    override fun release() {
        sessionPre?.close(); sessionTrans?.close(); sessionDec?.close(); ortEnv?.close()
    }
}