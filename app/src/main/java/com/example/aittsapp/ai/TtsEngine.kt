package com.example.aittsapp.ai

import android.content.Context

/**
 * Interfaz para el motor de síntesis de voz mediante IA.
 */
interface TtsEngine {
    fun initialize(context: Context)
    fun synthesize(text: String, profile: VoiceProfile): ByteArray?
    fun release()
}