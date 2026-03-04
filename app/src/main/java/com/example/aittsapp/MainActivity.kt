package com.example.aittsapp

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.MediaRecorder
import android.net.Uri
import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ListView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.lifecycle.lifecycleScope
import com.example.aittsapp.ai.VoiceCloner
import com.example.aittsapp.ai.VoiceManager
import com.example.aittsapp.ai.VoiceProfile
import kotlinx.coroutines.launch
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var voiceManager: VoiceManager
    private lateinit var voiceCloner: VoiceCloner
    
    private var recorder: MediaRecorder? = null
    private var audioFile: File? = null
    private var isRecording = false

    private lateinit var adapter: ArrayAdapter<String>
    private val voiceNames = mutableListOf<String>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        voiceManager = VoiceManager(this)
        voiceCloner = VoiceCloner(this)

        setupUI()
        refreshVoiceList()
        checkPermissions()
    }

    private fun setupUI() {
        val btnRecord = findViewById<Button>(R.id.btnRecordVoice)
        val btnProcess = findViewById<Button>(R.id.btnProcessCloning)
        
        // 1. Lógica de Micrófono (Grabar 30s)
        btnRecord.setOnClickListener {
            if (!isRecording) {
                startRecording()
                btnRecord.text = "Detener Grabación (30s máx)"
            } else {
                stopRecording()
                btnRecord.text = "Grabar/Subir 30s de audio"
            }
        }

        // 2. Lógica de Selección de Archivo
        val btnPickFile = Button(this).apply {
            text = "Seleccionar archivo de audio"
            setOnClickListener { filePickerLauncher.launch("audio/*") }
        }
        (findViewById<android.view.ViewGroup>(android.R.id.content).getChildAt(0) as? android.widget.LinearLayout)?.addView(btnPickFile, 2)

        // 3. Procesar Clonación
        btnProcess.setOnClickListener {
            if (audioFile == null || !audioFile!!.exists()) {
                Toast.makeText(this, "Primero graba o sube un audio", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            processVoiceCloning()
        }

        // Lista de voces
        val listView = ListView(this)
        adapter = ArrayAdapter(this, android.R.layout.simple_list_item_1, voiceNames)
        listView.adapter = adapter
        (findViewById<android.view.ViewGroup>(android.R.id.content).getChildAt(0) as? android.widget.LinearLayout)?.addView(listView)
    }

    private val filePickerLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        uri?.let {
            val tempFile = File(cacheDir, "picked_voice.wav")
            contentResolver.openInputStream(it)?.use { input ->
                tempFile.outputStream().use { output -> input.copyTo(output) }
            }
            audioFile = tempFile
            Toast.makeText(this, "Audio cargado correctamente", Toast.LENGTH_SHORT).show()
        }
    }

    private fun startRecording() {
        isRecording = true
        audioFile = File(cacheDir, "recorded_voice.amr")
        recorder = MediaRecorder().apply {
            setAudioSource(MediaRecorder.AudioSource.MIC)
            setOutputFormat(MediaRecorder.OutputFormat.AMR_WB)
            setAudioEncoder(MediaRecorder.AudioEncoder.AMR_WB)
            setOutputFile(audioFile!!.absolutePath)
            prepare()
            start()
        }
        Toast.makeText(this, "Grabando...", Toast.LENGTH_SHORT).show()
    }

    private fun stopRecording() {
        isRecording = false
        recorder?.apply {
            stop()
            release()
        }
        recorder = null
        Toast.makeText(this, "Grabación guardada", Toast.LENGTH_SHORT).show()
    }

    private fun processVoiceCloning() {
        lifecycleScope.launch {
            Toast.makeText(this@MainActivity, "IA analizando voz (clonación zero-shot)...", Toast.LENGTH_LONG).show()
            val embedding = voiceCloner.cloneFromAudio(audioFile!!)
            if (embedding != null) {
                val profile = voiceManager.saveClonedVoice(embedding)
                Toast.makeText(this@MainActivity, "¡Voz '\${profile.name}' agregada!", Toast.LENGTH_LONG).show()
                refreshVoiceList()
            }
        }
    }

    private fun refreshVoiceList() {
        val allVoices = voiceManager.getDefaultVoices() + voiceManager.getClonedVoices()
        voiceNames.clear()
        allVoices.forEach { voiceNames.add("\${it.name} (\${if (it.isCloned) "Clonada" else "Base"})") }
        adapter.notifyDataSetChanged()
    }

    private fun checkPermissions() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), 100)
        }
    }
}