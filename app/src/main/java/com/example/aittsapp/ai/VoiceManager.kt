package com.example.aittsapp.ai

import android.content.Context
import java.io.File
import java.util.UUID

class VoiceManager(private val context: Context) {
    
    private val clonedDir = File(context.filesDir, "cloned_voices").apply { mkdirs() }

    fun getDefaultVoices(): List<VoiceProfile> {
        return listOf(
            VoiceProfile("cl-female-base", "Mujer Chilena (IA)", "es-CL", Gender.FEMALE),
            VoiceProfile("cl-male-base", "Hombre Chileno (IA)", "es-CL", Gender.MALE)
        )
    }

    fun getClonedVoices(): List<VoiceProfile> {
        return clonedDir.listFiles()?.filter { it.extension == "wav" }?.mapNotNull { audioFile ->
            val parts = audioFile.nameWithoutExtension.split("_")
            if (parts.size >= 3) {
                val id = parts[0]
                val gender = if (parts[1] == "male") Gender.MALE else Gender.FEMALE
                val name = parts[2]
                val textFile = File(clonedDir, "${audioFile.nameWithoutExtension}.txt")
                val refText = if (textFile.exists()) textFile.readText() else ""
                VoiceProfile(id, name, "es-CL", gender, true, audioFile, refText)
            } else null
        } ?: emptyList()
    }

    fun saveClonedVoice(embedding: FloatArray, customName: String, gender: Gender, refText: String): VoiceProfile {
        val id = UUID.randomUUID().toString()
        val genderStr = if (gender == Gender.MALE) "male" else "female"
        val baseName = "${id}_${genderStr}_${customName}"
        
        // Guardamos el audio real para enviarlo al servidor
        val audioDest = File(clonedDir, "$baseName.wav")
        // En una implementación real, aquí moveríamos el archivo temporal al destino final
        
        // Guardamos el texto de referencia
        val textFile = File(clonedDir, "$baseName.txt")
        textFile.writeText(refText)
        
        return VoiceProfile(id, customName, "es-CL", gender, true, audioDest, refText)
    }

    fun deleteVoice(profile: VoiceProfile): Boolean {
        val baseName = profile.referenceAudio?.nameWithoutExtension ?: return false
        val audioFile = File(clonedDir, "$baseName.wav")
        val textFile = File(clonedDir, "$baseName.txt")
        return audioFile.delete() && textFile.delete()
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