package com.example.aittsapp.engine

/**
 * Pre-procesador avanzado para el español de Chile.
 * Ajusta el texto para que la IA emule la fonética regional.
 */
object ChileanPreProcessor {
    
    fun process(text: String): String {
        var p = text.lowercase()
        
        // 1. Manejo de pausas y énfasis (Prosodia)
        p = p.replace("?", "? ")
        p = p.replace("!", "! ")
        p = p.replace(",", ", ")
        
        // 2. Fonética Chilena (Ajustes para el Transformer)
        // La IA lee mejor si el texto refleja la intención sonora
        val phoneticRules = mapOf(
            "chile" to "chiile",
            "estás" to "estái",
            "estamos" to "estamo",
            "bueno" to "güeno",
            "verdad" to "verdáh",
            "entonces" to "entonce",
            "asado" to "asao",
            "pescado" to "pescao",
            "complicado" to "complicao"
        )
        
        phoneticRules.forEach { (old, new) ->
            p = p.replace(Regex("\\b$old\\b"), new)
        }
        
        // 3. Modismos comunes
        p = p.replace("cachai", "cachaai")
        p = p.replace("fome", "foome")
        p = p.replace("bacán", "bacaan")
        
        return p.trim()
    }
}