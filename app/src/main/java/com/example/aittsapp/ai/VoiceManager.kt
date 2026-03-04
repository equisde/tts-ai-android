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

    fun saveClonedVoice(audioSource: File, customName: String, gender: Gender, refText: String): VoiceProfile {
        val id = UUID.randomUUID().toString()
        val genderStr = if (gender == Gender.MALE) "male" else "female"
        val baseName = "${id}_${genderStr}_${customName}"
        
        // FIX: Mover físicamente el archivo de audio a la carpeta permanente
        val audioDest = File(clonedDir, "$baseName.wav")
        audioSource.copyTo(audioDest, overwrite = true)
        
        // Guardar el texto de referencia
        val textFile = File(clonedDir, "$baseName.txt")
        textFile.writeText(refText)
        
        return VoiceProfile(id, customName, "es-CL", gender, true, audioDest, refText)
    }

    fun deleteVoice(profile: VoiceProfile): Boolean {
        val audioFile = profile.referenceAudio ?: return false
        val baseName = audioFile.nameWithoutExtension
        val textFile = File(clonedDir, "$baseName.txt")
        return audioFile.delete() && textFile.delete()
    }
}