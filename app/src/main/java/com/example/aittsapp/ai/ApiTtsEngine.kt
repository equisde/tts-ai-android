package com.example.aittsapp.ai

import android.content.Context
import android.util.Log
import com.example.aittsapp.engine.LogManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.asRequestBody
import org.json.JSONObject
import java.io.File
import java.util.concurrent.TimeUnit

class ApiTtsEngine : TtsEngine {
    private val TAG = "ApiTtsEngine"
    private lateinit var client: OkHttpClient
    private var serverUrl: String = "http://52.1.236.194:8000"

    override fun initialize(context: Context) {
        val prefs = context.getSharedPreferences("TTS_PREFS", Context.MODE_PRIVATE)
        serverUrl = prefs.getString("SERVER_URL", "http://52.1.236.194:8000") ?: "http://52.1.236.194:8000"
        
        client = OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .writeTimeout(10, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .build()
            
        LogManager.log("Cliente API iniciado apuntando a: " + serverUrl)
    }

    override fun synthesize(text: String, profile: VoiceProfile): ByteArray? {
        // ERROR FIX: Las peticiones de red NO pueden ir en el hilo principal
        LogManager.log("Solicitando síntesis para: " + text.take(15) + "...")
        
        try {
            val json = JSONObject().apply {
                put("text", text)
                put("reference_text", profile.referenceText)
                put("gender", if (profile.gender == Gender.MALE) "male" else "female")
            }

            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("data", json.toString())

            if (profile.referenceAudio != null && profile.referenceAudio.exists()) {
                requestBody.addFormDataPart(
                    "audio", 
                    profile.referenceAudio.name,
                    profile.referenceAudio.asRequestBody("audio/*".toMediaType())
                )
            }

            val request = Request.Builder()
                .url(serverUrl + "/synthesize")
                .post(requestBody.build())
                .build()

            val response = client.newCall(request).execute()
            if (response.isSuccessful) {
                return response.body?.bytes()
            } else {
                LogManager.log("Error Servidor: " + response.code)
            }
        } catch (e: Exception) {
            LogManager.log("Error Red: " + e.message)
        }
        return null
    }

    override fun release() {}
}