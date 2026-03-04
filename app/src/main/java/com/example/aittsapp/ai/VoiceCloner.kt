package com.example.aittsapp.ai

import android.util.Log
import java.io.File

class VoiceCloner(private val context: android.content.Context) {
    private val TAG = "VoiceCloner"
    private val ttsEngine = ApiTtsEngine()

    init {
        ttsEngine.initialize(context)
    }

    suspend fun cloneFromAudio(audioFile: File): FloatArray? = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.Default) {
        // Obsoleto: Usamos cloneRemote directamente
        FloatArray(256) { 0f }
    }

    suspend fun cloneRemote(profile: VoiceProfile): Boolean {
        return ttsEngine.clone(profile)
    }

    suspend fun synthesizeForTest(text: String, profile: VoiceProfile): ByteArray? {
        return ttsEngine.synthesize(text, profile)
    }
}