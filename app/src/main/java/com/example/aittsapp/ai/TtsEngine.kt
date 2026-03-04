package com.example.aittsapp.ai

import android.content.Context
import java.io.File

/**
 * Representa un perfil vocal (Base o Clonado).
 */
data class VoiceProfile(
    val id: String,
    val name: String,
    val language: String = "es-CL",
    val gender: Gender,
    val isCloned: Boolean = false,
    val embeddingFile: File? = null // El resultado de los 30s de audio
)

enum class Gender { MALE, FEMALE }

/**
 * Interfaz para el motor de síntesis de voz mediante IA.
 */
interface TtsEngine {
    fun initialize(context: Context)
    fun synthesize(text: String, profile: VoiceProfile): ByteArray?
    fun release()
}