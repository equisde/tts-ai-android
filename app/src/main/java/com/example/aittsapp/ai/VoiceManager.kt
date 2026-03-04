package com.example.aittsapp.ai

import android.content.Context
import android.util.Log
import java.io.File
import java.util.UUID

/**
 * Gestor de perfiles de voz.
 * Mantiene las voces predefinidas (CL-Hombre/Mujer) y las clonadas.
 */
class VoiceManager(private val context: Context) {
    
    private val clonedDir = File(context.filesDir, "cloned_voices").apply { mkdirs() }

    fun getDefaultVoices(): List<VoiceProfile> {
        return listOf(
            VoiceProfile("cl-female-base", "Mujer Chilena (IA)", "es-CL", Gender.FEMALE),
            VoiceProfile("cl-male-base", "Hombre Chileno (IA)", "es-CL", Gender.MALE)
        )
    }

    fun getClonedVoices(): List<VoiceProfile> {
        return clonedDir.listFiles()?.map { file ->
            VoiceProfile(
                id = file.nameWithoutExtension,
                name = "Voz Clonada - \${file.nameWithoutExtension.take(4)}",
                gender = Gender.FEMALE, // Podría inferirse
                isCloned = true,
                referenceAudio = file,
                referenceText = ""
            )
        } ?: emptyList()
    }

    fun saveClonedVoice(embedding: FloatArray): VoiceProfile {
        val id = UUID.randomUUID().toString()
        val file = File(clonedDir, "\$id.bin")
        // Guardar los bytes del embedding para uso futuro
        file.writeBytes(floatArrayToByteArray(embedding))
        
        return VoiceProfile(id, "Nueva Voz Clonada", "es-CL", Gender.FEMALE, true, file, "")
    }

    private fun floatArrayToByteArray(floats: FloatArray): ByteArray {
        val bytes = ByteArray(floats.size * 4)
        for (i in floats.indices) {
            val intBits = java.lang.Float.floatToIntBits(floats[i])
            bytes[i * 4] = (intBits and 0xFF).toByte()
            bytes[i * 4 + 1] = ((intBits shr 8) and 0xFF).toByte()
            bytes[i * 4 + 2] = ((intBits shr 16) and 0xFF).toByte()
            bytes[i * 4 + 3] = ((intBits shr 24) and 0xFF).toByte()
        }
        return bytes
    }
}