package com.example.aittsapp.ai

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Clase encargada de la clonación de voz "Zero-Shot".
 * Utiliza un modelo Speaker Encoder (IA) para procesar 30s de audio.
 */
class VoiceCloner(private val context: android.content.Context) {
    private val TAG = "VoiceCloner"
    private val ttsEngine = ApiTtsEngine()

    init {
        ttsEngine.initialize(context)
    }

    /**
     * Devuelve un embedding simulado, ya que el audio real se enviará al servidor
     * para clonación "Zero-Shot" en cada síntesis.
     */
    suspend fun cloneFromAudio(audioFile: File): FloatArray? = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Default) {
        Log.i(TAG, "Guardando perfil para: \${audioFile.absolutePath}")
        FloatArray(256) { (Math.random() * 2 - 1).toFloat() }
    }

    fun synthesizeForTest(text: String, profile: VoiceProfile): ByteArray? {
        return ttsEngine.synthesize(text, profile)
    }
}