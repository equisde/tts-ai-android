package com.example.aittsapp.ai

import java.io.File

/**
 * Representa un perfil vocal real para F5-TTS.
 * Requiere un audio de referencia y su transcripción para la clonación zero-shot.
 */
data class VoiceProfile(
    val id: String,
    val name: String,
    val language: String = "es-CL",
    val gender: Gender,
    val isCloned: Boolean = false,
    val referenceAudio: File? = null,
    val referenceText: String = "" // Texto que se dice en el audio de referencia
)

enum class Gender { MALE, FEMALE }