package com.example.aittsapp

import android.media.AudioFormat
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.speech.tts.Voice
import android.util.Log
import com.example.aittsapp.ai.ApiTtsEngine
import com.example.aittsapp.ai.TtsEngine
import com.example.aittsapp.ai.VoiceManager
import com.example.aittsapp.ai.VoiceProfile
import com.example.aittsapp.ai.Gender
import com.example.aittsapp.engine.ChileanPreProcessor
import java.util.Locale

class TtsEngineService : TextToSpeechService() {

    private val TAG = "AITtsEngineService"
    private lateinit var ttsEngine: TtsEngine
    private lateinit var voiceManager: VoiceManager
    
    override fun onCreate() {
        super.onCreate()
        voiceManager = VoiceManager(applicationContext)
        Thread {
            try {
                ttsEngine = ApiTtsEngine()
                ttsEngine.initialize(applicationContext)
                Log.i(TAG, "Motor API inicializado.")
            } catch (e: Exception) {
                Log.e(TAG, "Error crítico al inicializar IA API", e)
            }
        }.start()
    }

    override fun onIsLanguageAvailable(lang: String?, country: String?, variant: String?): Int {
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

    /**
     * Reporta las voces reales al sistema Android.
     */
    override fun onGetVoices(): MutableList<Voice> {
        val voices = mutableListOf<Voice>()
        val allProfiles = voiceManager.getDefaultVoices() + voiceManager.getClonedVoices()
        
        allProfiles.forEach { profile ->
            val locale = Locale("es", "CL")
            val features = mutableSetOf<String>()
            if (profile.isCloned) features.add("cloned_voice") 
            
            voices.add(Voice(
                profile.id,
                locale,
                Voice.QUALITY_VERY_HIGH,
                Voice.LATENCY_NORMAL,
                false,
                features
            ))
        }
        return voices
    }

    override fun onIsValidVoiceName(voiceName: String?): Int {
        val all = voiceManager.getDefaultVoices() + voiceManager.getClonedVoices()
        return if (all.any { it.id == voiceName }) TextToSpeech.SUCCESS else TextToSpeech.ERROR
    }

    override fun onStop() {
        Log.i(TAG, "Parada solicitada por el sistema.")
    }

    override fun onDestroy() {
        super.onDestroy()
        ttsEngine.release()
    }

    override fun onSynthesizeText(request: SynthesisRequest?, callback: SynthesisCallback?) {
        if (request == null || callback == null) return
        
        // 1. Pre-procesar el texto para naturalidad chilena
        val rawText = request.charSequenceText.toString()
        val processedText = ChileanPreProcessor.process(rawText)
        
        // 2. Obtener configuración de voz (ajustada por el usuario en el sistema)
        val voiceName = request.voiceName ?: getSharedPreferences("TTS_PREFS", MODE_PRIVATE).getString("DEFAULT_VOICE_ID", "cl-female-base")
        val pitch = request.pitch // El sistema puede pedir cambio de tono
        val speechRate = request.speechRate // El sistema puede pedir cambio de velocidad

        val allVoices = voiceManager.getDefaultVoices() + voiceManager.getClonedVoices()
        val selectedProfile = allVoices.find { it.id == voiceName } ?: allVoices[0]

        // 3. Informar al sistema de la configuración de audio
        val sampleRate = 24000
        callback.start(sampleRate, AudioFormat.ENCODING_PCM_16BIT, 1)

        try {
            // 4. Inferencia por fragmentos en segundo plano para evitar ANR (App Not Responding)
            Thread {
                try {
                    val sentences = processedText.split(Regex("(?<=[.!?])\\s+"))
                    sentences.forEach { sentence ->
                        val pcmData = ttsEngine.synthesize(sentence, selectedProfile)
                        if (pcmData != null) {
                            callback.audioAvailable(pcmData, 0, pcmData.size)
                        }
                    }
                    callback.done()
                } catch (innerE: Exception) {
                    Log.e(TAG, "Error en hilo de síntesis", innerE)
                    callback.error()
                }
            }.start()
        } catch (e: Exception) {
            Log.e(TAG, "Error al lanzar hilo de síntesis", e)
            callback.error()
        }
    }
}