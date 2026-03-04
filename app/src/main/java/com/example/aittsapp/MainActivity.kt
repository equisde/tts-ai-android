package com.example.aittsapp

import android.content.Intent
import android.media.MediaRecorder
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.aittsapp.ai.VoiceCloner
import com.example.aittsapp.ai.VoiceManager
import com.example.aittsapp.ai.VoiceProfile
import com.example.aittsapp.audio.AudioTestPlayer
import com.example.aittsapp.engine.PermissionsManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var voiceManager: VoiceManager
    private lateinit var voiceCloner: VoiceCloner
    private lateinit var permissionsManager: PermissionsManager
    private val audioPlayer = AudioTestPlayer()
    
    private var recorder: MediaRecorder? = null
    private var audioFile: File? = null
    private var isRecording = false

    private lateinit var adapter: ArrayAdapter<String>
    private val voiceNames = mutableListOf<String>()
    private var selectedVoiceIndex = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        voiceManager = VoiceManager(this)
        voiceCloner = VoiceCloner(this)
        permissionsManager = PermissionsManager(this)

        setupUI()
        refreshVoiceList()
        
        if (!permissionsManager.hasAllPermissions()) {
            permissionsManager.requestPermissions(PermissionsManager.PERMISSIONS_REQUEST_CODE)
        }
    }

    private fun setupUI() {
        val btnRecord = findViewById<Button>(R.id.btnRecordVoice)
        val btnPickFile = findViewById<Button>(R.id.btnPickFile)
        val btnProcess = findViewById<Button>(R.id.btnProcessCloning)
        val btnTestVoice = findViewById<Button>(R.id.btnTestVoice)
        val etTestText = findViewById<com.google.android.material.textfield.TextInputEditText>(R.id.etTestText)
        val btnSystemSettings = findViewById<Button>(R.id.btnOpenSystemSettings)
        val listView = findViewById<ListView>(R.id.listViewVoices)
        
        btnRecord.setOnClickListener {
            if (!isRecording) startRecording() else stopRecording()
        }

        btnPickFile.setOnClickListener { filePickerLauncher.launch("audio/*") }

        btnProcess.setOnClickListener {
            if (audioFile != null && audioFile!!.exists()) processVoiceCloning()
            else Toast.makeText(this, "Primero graba o sube un audio", Toast.LENGTH_SHORT).show()
        }

        btnTestVoice.setOnClickListener {
            val text = etTestText.text.toString()
            lifecycleScope.launch {
                val all = voiceManager.getDefaultVoices() + voiceManager.getClonedVoices()
                if (selectedVoiceIndex < all.size) {
                    Toast.makeText(this@MainActivity, "Generando audio...", Toast.LENGTH_SHORT).show()
                    val pcm = withContext(Dispatchers.Default) { 
                        voiceCloner.synthesizeForTest(text, all[selectedVoiceIndex]) 
                    }
                    pcm?.let { audioPlayer.playPcm(it) }
                }
            }
        }

        btnSystemSettings.setOnClickListener { startActivity(Intent("com.android.settings.TTS_SETTINGS")) }

        adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_list_item_single_choice, voiceNames)
        listView.adapter = adapter
        listView.choiceMode = ListView.CHOICE_MODE_SINGLE
        listView.setOnItemClickListener { _, _, position, _ -> selectedVoiceIndex = position }
    }

    private val filePickerLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            val tempFile = File(cacheDir, "picked_voice.wav")
            contentResolver.openInputStream(it)?.use { input ->
                tempFile.outputStream().use { output -> input.copyTo(output) }
            }
            audioFile = tempFile
            Toast.makeText(this, "Audio cargado", Toast.LENGTH_SHORT).show()
        }
    }

    private fun startRecording() {
        try {
            isRecording = true
            audioFile = File(cacheDir, "recorded_voice.amr")
            recorder = MediaRecorder().apply {
                setAudioSource(MediaRecorder.AudioSource.MIC)
                setOutputFormat(MediaRecorder.OutputFormat.AMR_WB)
                setAudioEncoder(MediaRecorder.AudioEncoder.AMR_WB)
                setOutputFile(audioFile!!.absolutePath)
                prepare(); start()
            }
            findViewById<Button>(R.id.btnRecordVoice).text = "Detener Grabación"
        } catch (e: Exception) { isRecording = false }
    }

    private fun stopRecording() {
        isRecording = false
        recorder?.apply { stop(); release() }
        recorder = null
        findViewById<Button>(R.id.btnRecordVoice).text = "Grabar/Subir 30s de audio"
    }

    private fun processVoiceCloning() {
        lifecycleScope.launch {
            Toast.makeText(this@MainActivity, "Analizando identidad vocal...", Toast.LENGTH_SHORT).show()
            val embedding = voiceCloner.cloneFromAudio(audioFile!!)
            if (embedding != null) {
                voiceManager.saveClonedVoice(embedding)
                refreshVoiceList()
                Toast.makeText(this@MainActivity, "¡Voz clonada añadida!", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun refreshVoiceList() {
        val allVoices = voiceManager.getDefaultVoices() + voiceManager.getClonedVoices()
        voiceNames.clear()
        allVoices.forEach { 
            val type = if (it.isCloned) "Clonada" else "Base"
            voiceNames.add("\${it.name} (\$type)") 
        }
        if (::adapter.isInitialized) adapter.notifyDataSetChanged()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PermissionsManager.PERMISSIONS_REQUEST_CODE && permissionsManager.hasAllPermissions()) {
            Toast.makeText(this, "Permisos listos", Toast.LENGTH_SHORT).show()
        }
    }
}