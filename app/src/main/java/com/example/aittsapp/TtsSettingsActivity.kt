package com.example.aittsapp

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class TtsSettingsActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val tv = TextView(this).apply {
            text = "Configuraciones del Motor TTS AI (Español Chileno)"
            textSize = 20f
            setPadding(32, 32, 32, 32)
        }
        setContentView(tv)
    }
}