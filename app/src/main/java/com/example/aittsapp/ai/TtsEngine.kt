package com.example.aittsapp.ai

import android.content.Context

interface TtsEngine {
    fun initialize(context: Context)
    suspend fun synthesize(text: String, profile: VoiceProfile): ByteArray?
    suspend fun clone(profile: VoiceProfile): Boolean
    fun release()
}