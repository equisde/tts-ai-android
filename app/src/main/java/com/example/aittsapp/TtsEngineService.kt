package com.example.aittsapp

import android.media.AudioFormat
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.util.Log
import com.example.aittsapp.ai.OnnxTtsEngine
import com.example.aittsapp.ai.TtsEngine
import com.example.aittsapp.ai.VoiceManager
import com.example.aittsapp.ai.VoiceProfile

class TtsEngineService : TextToSpeechService() {

    private val TAG = "AITtsEngineService"
    private lateinit var ttsEngine: TtsEngine
    private lateinit var voiceManager: VoiceManager
    
    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "Inicializando motor AI TTS LLM para Android 15")
        
        // Inicialización modular
        voiceManager = VoiceManager(applicationContext)
        ttsEngine = OnnxTtsEngine()
        ttsEngine.initialize(applicationContext)
    }

    override fun onIsLanguageAvailable(lang: String?, country: String?, variant: String?): Int {
        // Soporte estricto para Español Chileno
        if (lang == "spa" && (country == "CHL" || country == "CHL".lowercase())) {
            return TextToSpeech.LANG_COUNTRY_AVAILABLE
        }
        return TextToSpeech.LANG_NOT_SUPPORTED
    }

    override fun onGetLanguage(): Array<String> {
        return arrayOf("spa", "CHL", "")
    }

    override fun onLoadLanguage(lang: String?, country: String?, variant: String?): Int {
        return onIsLanguageAvailable(lang, country, variant)
    }

    override fun onStop() {
        Log.i(TAG, "Deteniendo síntesis de voz.")
    }

    override fun onDestroy() {
        super.onDestroy()
        ttsEngine.release()
    }

    override fun onSynthesizeText(request: SynthesisRequest?, callback: SynthesisCallback?) {
        if (request == null || callback == null) return
        
        val text = request.charSequenceText.toString()
        val voiceName = request.voiceName ?: "cl-female-base"

        // 1. Buscar el perfil de voz (Base o Clonada)
        val allVoices = voiceManager.getDefaultVoices() + voiceManager.getClonedVoices()
        val selectedProfile = allVoices.find { it.id == voiceName } ?: allVoices[0]

        // 2. Configuración de audio profesional (24kHz es estándar para IA de alta calidad)
        val sampleRate = 24000
        callback.start(sampleRate, AudioFormat.ENCODING_PCM_16BIT, 1)

        try {
            // 3. Ejecutar Inferencia de IA (Modelo LLM TTS)
            val pcmData = ttsEngine.synthesize(text, selectedProfile)
            
            if (pcmData != null) {
                // Enviar el audio generado al sistema Android
                callback.audioAvailable(pcmData, 0, pcmData.size)
                callback.done()
                Log.d(TAG, "Síntesis completada con éxito para perfil: \${selectedProfile.name}")
            } else {
                callback.error()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error crítico en síntesis", e)
            callback.error()
        }
    }
}