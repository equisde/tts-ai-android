package com.example.aittsapp.engine

/**
 * Pre-procesador especializado en el español de Chile.
 * Normaliza abreviaturas, modismos y ajusta la puntuación para la prosodia chilena.
 */
object ChileanPreProcessor {
    
    fun process(text: String): String {
        var processed = text.lowercase()
        
        // Normalización de puntuación para pausas naturales de la IA
        processed = processed.replace("...", "...")
        processed = processed.replace("?", "? ")
        processed = processed.replace("!", "! ")
        
        // Manejo de modismos básicos para mejorar la fonética de la IA
        // (Ajustamos el texto para que el modelo F5-Spanish lo entienda mejor)
        val replacements = mapOf(
            "cachai" to "cachaai",
            "po" to "poh",
            "chile" to "chiile",
            "bacán" to "bacaan"
        )
        
        replacements.forEach { (old, new) ->
            processed = processed.replace(old, new)
        }
        
        return processed.trim()
    }
}