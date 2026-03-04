package com.example.aittsapp.engine

import androidx.lifecycle.MutableLiveData

/**
 * Gestor de logs visuales para depuración en tiempo real.
 */
object LogManager {
    val logUpdates = MutableLiveData<String>()
    private val fullLog = StringBuilder()

    fun log(message: String) {
        val timestamp = java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.getDefault()).format(java.util.Date())
        val formattedMessage = "[\$timestamp] \$message
"
        fullLog.insert(0, formattedMessage) // Los más nuevos arriba
        logUpdates.postValue(fullLog.toString())
    }

    fun clear() {
        fullLog.setLength(0)
        logUpdates.postValue("")
    }
}