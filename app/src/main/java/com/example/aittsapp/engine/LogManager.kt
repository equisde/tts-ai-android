package com.example.aittsapp.engine

import androidx.lifecycle.MutableLiveData
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Gestor de logs visuales para depuración en tiempo real.
 */
object LogManager {
    val logUpdates = MutableLiveData<String>()
    private val fullLog = StringBuilder()

    fun log(message: String) {
        val timestamp = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())
        val formattedMessage = "[$timestamp] $message\n"
        fullLog.insert(0, formattedMessage) // Los más nuevos arriba
        logUpdates.postValue(fullLog.toString())
    }

    fun clear() {
        fullLog.setLength(0)
        logUpdates.postValue("")
    }
}