package com.example.aittsapp.ai

import android.content.Context
import java.io.File
import java.util.UUID

/**
 * Gestor de perfiles de voz mejorado.
 * Permite guardar, listar y eliminar voces clonadas con metadatos (Género).
 */
class VoiceManager(private val context: Context) {
    
    private val clonedDir = File(context.filesDir, "cloned_voices").apply { mkdirs() }

    fun getDefaultVoices(): List<VoiceProfile> {
        return listOf(
            VoiceProfile("cl-female-base", "Mujer Chilena (IA)", "es-CL", Gender.FEMALE),
            VoiceProfile("cl-male-base", "Hombre Chileno (IA)", "es-CL", Gender.MALE)
        )
    }

    /**
     * Recupera las voces clonadas. 
     * Formato de archivo esperado: id_gender_name.bin
     */
    fun getClonedVoices(): List<VoiceProfile> {
        return clonedDir.listFiles()?.mapNotNull { file ->
            val parts = file.nameWithoutExtension.split("_")
            if (parts.size >= 3) {
                val id = parts[0]
                val gender = if (parts[1].lowercase() == "male") Gender.MALE else Gender.FEMALE
                val name = parts[2]
                VoiceProfile(id, name, "es-CL", gender, true, file, "")
            } else null
        } ?: emptyList()
    }

    fun saveClonedVoice(embedding: FloatArray, customName: String, gender: Gender): VoiceProfile {
        val id = UUID.randomUUID().toString()
        val genderStr = if (gender == Gender.MALE) "male" else "female"
        // Guardamos metadatos en el nombre del archivo para simplicidad
        val fileName = "${id}_${genderStr}_${customName}.bin"
        val file = File(clonedDir, fileName)
        file.writeBytes(floatArrayToByteArray(embedding))
        
        return VoiceProfile(id, customName, "es-CL", gender, true, file, "")
    }

    fun deleteVoice(profile: VoiceProfile): Boolean {
        return profile.referenceAudio?.delete() ?: false
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