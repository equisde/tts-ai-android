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
    private var serverUrl: String = "http://YOUR_SERVER_IP:8000"

    override fun initialize(context: Context) {
        val prefs = context.getSharedPreferences("TTS_PREFS", Context.MODE_PRIVATE)
        serverUrl = prefs.getString("SERVER_URL", "http://YOUR_SERVER_IP:8000") ?: "http://YOUR_SERVER_IP:8000"
        
        client = OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .readTimeout(300, TimeUnit.SECONDS) // AUMENTADO A 5 MINUTOS
            .build()
            
        LogManager.log("Cliente listo (Timeout 5min): " + serverUrl)
    }

    override suspend fun clone(profile: VoiceProfile): Boolean = withContext(Dispatchers.IO) {
        LogManager.log("Clonando identidad vocal...")
        val audioFile = profile.referenceAudio ?: return@withContext false
        
        val json = JSONObject().apply {
            put("voice_id", profile.id)
            put("reference_text", profile.referenceText)
        }

        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("data", json.toString())
            .addFormDataPart("audio", audioFile.name, audioFile.asRequestBody("audio/*".toMediaType()))
            .build()

        val request = Request.Builder().url(serverUrl + "/clone").post(requestBody).build()

        try {
            val response = client.newCall(request).execute()
            return@withContext response.isSuccessful
        } catch (e: Exception) {
            LogManager.log("Error Red (Clonar): " + e.message)
        }
        return@withContext false
    }

    override suspend fun synthesize(text: String, profile: VoiceProfile): ByteArray? = withContext(Dispatchers.IO) {
        try {
            val json = JSONObject().apply {
                put("text", text)
                put("voice_id", profile.id)
            }

            val request = Request.Builder()
                .url(serverUrl + "/synthesize")
                .post(MultipartBody.Builder().setType(MultipartBody.FORM)
                    .addFormDataPart("data", json.toString()).build())
                .build()

            val response = client.newCall(request).execute()
            if (response.isSuccessful) return@withContext response.body?.bytes()
            else LogManager.log("Server Error: " + response.code)
        } catch (e: Exception) {
            LogManager.log("Error Red (Synthesize): " + e.message)
        }
        return@withContext null
    }

    override fun release() {}
}