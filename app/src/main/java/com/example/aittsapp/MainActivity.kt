package com.example.aittsapp

import android.content.Intent
import android.media.MediaRecorder
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.view.LayoutInflater
import android.view.View
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.aittsapp.ai.VoiceCloner
import com.example.aittsapp.ai.VoiceManager
import com.example.aittsapp.ai.VoiceProfile
import com.example.aittsapp.ai.Gender
import com.example.aittsapp.audio.AudioTestPlayer
import com.example.aittsapp.engine.PermissionsManager
import com.example.aittsapp.engine.LogManager
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
        
        LogManager.logUpdates.observe(this) { logs ->
            findViewById<TextView>(R.id.tvAiLogs).text = logs
        }
        
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
            if (audioFile != null && audioFile!!.exists()) {
                showCloningDialog()
            } else {
                Toast.makeText(this, "Primero graba o sube un audio", Toast.LENGTH_SHORT).show()
            }
        }

        btnTestVoice.setOnClickListener {
            val text = etTestText?.text.toString()
            if (text.isEmpty()) return@setOnClickListener
            
            lifecycleScope.launch {
                val all = voiceManager.getDefaultVoices() + voiceManager.getClonedVoices()
                if (selectedVoiceIndex >= 0 && selectedVoiceIndex < all.size) {
                    val profile = all[selectedVoiceIndex]
                    LogManager.log("Probando voz: " + profile.name)
                    
                    val pcm = withContext(Dispatchers.IO) { 
                        voiceCloner.synthesizeForTest(text, profile) 
                    }
                    
                    if (pcm != null && pcm.isNotEmpty()) {
                        audioPlayer.playPcm(pcm)
                    } else {
                        LogManager.log("ERROR: La IA no devolvió audio")
                    }
                }
            }
        }

        btnSystemSettings.setOnClickListener { 
            startActivity(Intent("com.android.settings.TTS_SETTINGS")) 
        }

        adapter = ArrayAdapter(this, android.R.layout.simple_list_item_single_choice, voiceNames)
        listView.adapter = adapter
        listView.choiceMode = ListView.CHOICE_MODE_SINGLE
        listView.setOnItemClickListener { _, _, position, _ -> selectedVoiceIndex = position }
        
        listView.setOnItemLongClickListener { _, _, position, _ ->
            val all = voiceManager.getDefaultVoices() + voiceManager.getClonedVoices()
            if (position < all.size) {
                val profile = all[position]
                if (profile.isCloned) showDeleteDialog(profile)
            }
            true
        }
    }

    private fun showCloningDialog() {
        val builder = AlertDialog.Builder(this)
        builder.setTitle("Clonación Inteligente")
        
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(50, 20, 50, 20)
        }

        val inputName = EditText(this).apply { hint = "Nombre de la voz" }
        container.addView(inputName)

        val inputRefText = EditText(this).apply { 
            hint = "¿Qué dice el audio de 30s? (Crucial)" 
            minLines = 2
        }
        container.addView(inputRefText)

        val labelGender = TextView(this).apply {
            text = "Género de la voz:"
            setPadding(0, 20, 0, 10)
        }
        container.addView(labelGender)

        val rgGender = RadioGroup(this)
        val rbFemale = RadioButton(this).apply { text = "Mujer"; id = View.generateViewId(); isChecked = true }
        val rbMale = RadioButton(this).apply { text = "Hombre"; id = View.generateViewId() }
        rgGender.addView(rbFemale); rgGender.addView(rbMale)
        container.addView(rgGender)

        builder.setView(container)

        builder.setPositiveButton("Clonar") { _, _ ->
            val name = inputName.text.toString().ifEmpty { "Voz Clonada" }
            val refText = inputRefText.text.toString()
            val gender = if (rbMale.isChecked) Gender.MALE else Gender.FEMALE
            processVoiceCloning(name, gender, refText)
        }
        builder.setNegativeButton("Cancelar", null).show()
    }

    private fun showDeleteDialog(profile: VoiceProfile) {
        AlertDialog.Builder(this)
            .setTitle("Eliminar voz")
            .setMessage("¿Eliminar '" + profile.name + "'?")
            .setPositiveButton("Eliminar") { _, _ ->
                if (voiceManager.deleteVoice(profile)) {
                    refreshVoiceList()
                }
            }
            .setNegativeButton("Cancelar", null).show()
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
        findViewById<Button>(R.id.btnRecordVoice).text = "Grabar Micrófono"
    }

    private fun processVoiceCloning(customName: String, gender: Gender, refText: String) {
        lifecycleScope.launch {
            LogManager.log("Iniciando clonación para: " + customName)
            val embedding = voiceCloner.cloneFromAudio(audioFile!!)
            if (embedding != null) {
                // Ahora VoiceManager guarda también el texto de referencia
                voiceManager.saveClonedVoice(embedding, customName, gender, refText)
                refreshVoiceList()
                LogManager.log("¡Voz '" + customName + "' guardada con éxito!")
            }
        }
    }

    private fun refreshVoiceList() {
        val allVoices = voiceManager.getDefaultVoices() + voiceManager.getClonedVoices()
        voiceNames.clear()
        for (profile in allVoices) {
            val typeStr = if (profile.isCloned) "Clonada" else "Base"
            val genderStr = if (profile.gender == Gender.MALE) "Hombre" else "Mujer"
            voiceNames.add(profile.name + " (" + genderStr + " - " + typeStr + ")")
        }
        if (::adapter.isInitialized) {
            adapter.notifyDataSetChanged()
            findViewById<ListView>(R.id.listViewVoices).setItemChecked(selectedVoiceIndex, true)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PermissionsManager.PERMISSIONS_REQUEST_CODE && permissionsManager.hasAllPermissions()) {
            Toast.makeText(this, "Permisos listos", Toast.LENGTH_SHORT).show()
        }
    }
}