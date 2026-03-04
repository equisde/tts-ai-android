package com.example.aittsapp

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.aittsapp.ai.VoiceManager
import com.google.android.material.button.MaterialButton

class TtsSettingsActivity : AppCompatActivity() {

    private lateinit var voiceManager: VoiceManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_tts_settings)

        voiceManager = VoiceManager(this)
        
        val rgVoices = findViewById<RadioGroup>(R.id.rgVoices)
        val btnManage = findViewById<MaterialButton>(R.id.btnManageVoices)
        
        val sharedPrefs = getSharedPreferences("TTS_PREFS", Context.MODE_PRIVATE)
        val selectedVoiceId = sharedPrefs.getString("DEFAULT_VOICE_ID", "cl-female-base")

        val allVoices = voiceManager.getDefaultVoices() + voiceManager.getClonedVoices()
        
        allVoices.forEach { profile ->
            val rb = RadioButton(this).apply {
                val type = if (profile.isCloned) "Clonada" else "Base"
                text = "\${profile.name} (\$type)"
                id = profile.id.hashCode() // ID único
                tag = profile.id
                setTextColor(getColor(android.R.color.white))
            }
            rgVoices.addView(rb)
            
            if (profile.id == selectedVoiceId) {
                rgVoices.check(rb.id)
            }
        }

        rgVoices.setOnCheckedChangeListener { group, checkedId ->
            val rb = group.findViewById<RadioButton>(checkedId)
            val newVoiceId = rb.tag as String
            
            sharedPrefs.edit().putString("DEFAULT_VOICE_ID", newVoiceId).apply()
            Toast.makeText(this, "Voz del sistema actualizada", Toast.LENGTH_SHORT).show()
        }

        btnManage.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
            finish()
        }
    }
}