package com.example.aittsapp.ai

import android.content.Context
import android.util.Log
import com.example.aittsapp.engine.LogManager
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

class ApiTtsEngine : TtsEngine {
    private val TAG = "ApiTtsEngine"
    private lateinit var client: OkHttpClient
    private var serverUrl: String = "http://YOUR_SERVER_IP:8000" // IP de servidor remoto

    override fun initialize(context: Context) {
        val prefs = context.getSharedPreferences("TTS_PREFS", Context.MODE_PRIVATE)
        serverUrl = prefs.getString("SERVER_URL", "http://192.168.1.100:8000") ?: "http://192.168.1.100:8000"
        
        client = OkHttpClient.Builder()
            .connectTimeout(15, TimeUnit.SECONDS)
            .writeTimeout(15, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS) // La síntesis puede tardar
            .build()
            
        LogManager.log("Cliente API iniciado apuntando a: \$serverUrl")
    }

    override fun synthesize(text: String, profile: VoiceProfile): ByteArray? {
        LogManager.log("Solicitando síntesis al servidor...")
        
        val json = JSONObject().apply {
            put("text", text)
            put("reference_text", profile.referenceText)
            put("gender", if (profile.gender == Gender.MALE) "male" else "female")
        }

        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("data", json.toString())

        // Adjuntar audio de referencia si existe
        if (profile.referenceAudio != null && profile.referenceAudio.exists()) {
            requestBody.addFormDataPart(
                "audio", 
                profile.referenceAudio.name,
                profile.referenceAudio.asRequestBody("audio/*".toMediaType())
            )
        }

        val request = Request.Builder()
            .url("\$serverUrl/synthesize")
            .post(requestBody.build())
            .build()

        try {
            val response = client.newCall(request).execute()
            if (response.isSuccessful) {
                LogManager.log("Audio recibido del servidor con éxito.")
                return response.body?.bytes()
            } else {
                LogManager.log("Error del servidor: \${response.code}")
            }
        } catch (e: Exception) {
            LogManager.log("Error de conexión: \${e.message}")
        }
        return null
    }

    override fun release() {
        // Limpiar recursos si es necesario
    }
}