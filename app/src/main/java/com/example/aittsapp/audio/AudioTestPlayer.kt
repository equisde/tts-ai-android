package com.example.aittsapp.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log

/**
 * Reproductor de audio de alta fidelidad optimizado para TTS de 24kHz.
 */
class AudioTestPlayer {
    private val SAMPLE_RATE = 24000
    private var audioTrack: AudioTrack? = null

    fun playPcm(data: ByteArray) {
        try {
            val minBufferSize = AudioTrack.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_OUT_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            )

            // Limpiar pista anterior si existe
            stop()

            audioTrack = AudioTrack.Builder()
                .setAudioAttributes(AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build())
                .setAudioFormat(AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build())
                .setBufferSizeInBytes(data.size.coerceAtLeast(minBufferSize))
                .setTransferMode(AudioTrack.MODE_STATIC) // Modo estático para frases cortas (mejor calidad)
                .build()

            audioTrack?.let {
                it.write(data, 0, data.size)
                it.play()
            }
        } catch (e: Exception) {
            Log.e("AudioTestPlayer", "Error al reproducir audio: \${e.message}")
        }
    }

    fun stop() {
        try {
            audioTrack?.apply {
                if (playState == AudioTrack.PLAYSTATE_PLAYING) {
                    stop()
                }
                release()
            }
            audioTrack = null
        } catch (e: Exception) {}
    }
}