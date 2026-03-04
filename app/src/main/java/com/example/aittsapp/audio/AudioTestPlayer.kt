package com.example.aittsapp.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack

/**
 * Reproductor real para pruebas de audio PCM 24kHz.
 */
class AudioTestPlayer {
    fun playPcm(data: ByteArray) {
        val bufferSize = AudioTrack.getMinBufferSize(24000, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val audioTrack = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_ASSISTANCE_SONIFICATION)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build())
            .setAudioFormat(AudioFormat.Builder()
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .setSampleRate(24000)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .build())
            .setBufferSizeInBytes(data.size.coerceAtLeast(bufferSize))
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()

        audioTrack.write(data, 0, data.size)
        audioTrack.play()
    }
}