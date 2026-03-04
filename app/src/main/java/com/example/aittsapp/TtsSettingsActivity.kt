package com.example.aittsapp

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.widget.EditText
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.aittsapp.ai.VoiceManager
import com.example.aittsapp.ai.Gender
import com.google.android.material.button.MaterialButton

class TtsSettingsActivity : AppCompatActivity() {

    private lateinit var voiceManager: VoiceManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_tts_settings)

        voiceManager = VoiceManager(this)
        
        val sharedPrefs = getSharedPreferences("TTS_PREFS", Context.MODE_PRIVATE)
        
        val etServerUrl = findViewById<EditText>(R.id.etServerUrl)
        val btnSaveServer = findViewById<MaterialButton>(R.id.btnSaveServer)
        
        etServerUrl.setText(sharedPrefs.getString("SERVER_URL", "http://YOUR_SERVER_IP:8000"))
        
        btnSaveServer.setOnClickListener {
            sharedPrefs.edit().putString("SERVER_URL", etServerUrl.text.toString()).apply()
            Toast.makeText(this, "Servidor guardado. Reinicia la app para aplicar.", Toast.LENGTH_LONG).show()
        }

        val rgVoices = findViewById<RadioGroup>(R.id.rgVoices)
        val btnManage = findViewById<MaterialButton>(R.id.btnManageVoices)
        
        val selectedVoiceId = sharedPrefs.getString("DEFAULT_VOICE_ID", "cl-female-base")
        
        val swHighQuality = findViewById<com.google.android.material.switchmaterial.SwitchMaterial>(R.id.swHighQuality)
        swHighQuality.isChecked = sharedPrefs.getBoolean("HIGH_QUALITY", false)
        swHighQuality.setOnCheckedChangeListener { _, isChecked ->
            sharedPrefs.edit().putBoolean("HIGH_QUALITY", isChecked).apply()
            val msg = if (isChecked) "Modo Alta Calidad (32 pasos)" else "Modo Turbo (16 pasos)"
            Toast.makeText(this, msg, Toast.LENGTH_SHORT).show()
        }

        val allVoices = voiceManager.getDefaultVoices() + voiceManager.getClonedVoices()
        
        allVoices.forEach { profile ->
            val rb = RadioButton(this).apply {
                val type = if (profile.isCloned) "Clonada" else "Base"
                val genderStr = if (profile.gender == Gender.MALE) "Hombre" else "Mujer"
                text = "${profile.name} ($genderStr - $type)"
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