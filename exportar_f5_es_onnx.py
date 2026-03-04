import os
import urllib.request
import zipfile
import shutil
import subprocess
import sys

def main():
    print("=== Preparando el entorno para exportar F5-TTS (Español Chileno) a ONNX ===")
    
    # Rutas detectadas
    extract_dir = "F5-TTS-ONNX-main"
    export_folder = os.path.join(extract_dir, "Export_ONNX", "F5_TTS")
    export_script_path = os.path.join(export_folder, "Export_F5.py")

    # 1. Ya instalamos dependencias en el paso anterior, pero nos aseguramos
    print("\n[1/4] Verificando dependencias...")

    # 2. El modelo ya está en F5_Spanish_Model
    print("\n[2/4] Verificando modelo F5-Spanish...")
    if not os.path.exists("F5_Spanish_Model/model_1200000.safetensors"):
        print("Error: No se encontró el modelo. Re-descargando...")
        from huggingface_hub import hf_hub_download
        os.makedirs("F5_Spanish_Model", exist_ok=True)
        hf_hub_download(repo_id="jpgallegoar/F5-Spanish", filename="model_1200000.safetensors", local_dir="F5_Spanish_Model")
        hf_hub_download(repo_id="jpgallegoar/F5-Spanish", filename="vocab.txt", local_dir="F5_Spanish_Model")

    # 3. Preparar archivos para la exportación en la carpeta correcta
    print("\n[3/4] Preparando archivos en la carpeta de exportación...")
    shutil.copy("F5_Spanish_Model/model_1200000.safetensors", os.path.join(export_folder, "model_1200000.safetensors"))
    shutil.copy("F5_Spanish_Model/vocab.txt", os.path.join(export_folder, "vocab.txt"))

    # 4. Modificar Export_F5.py para que use el modelo español local por defecto
    print("\n[4/4] Configurando script de exportación...")
    with open(export_script_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Ajustamos las rutas en el script para que no intente descargar el modelo inglés por defecto
    content = content.replace('checkpoint_path = "model_1200000.safetensors"', 'checkpoint_path = "model_1200000.safetensors"') # Ya está así
    content = content.replace('vocab_path = "vocab.txt"', 'vocab_path = "vocab.txt"') # Ya está así
    
    # Guardamos los cambios
    with open(export_script_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("\n====================================================================")
    print("¡TODO LISTO PARA LA EXPORTACIÓN FINAL!")
    print("Ejecuta los siguientes comandos para generar el modelo ONNX:")
    print(f"cd {export_folder}")
    print("python Export_F5.py")
    print("\nNota: Esto generará un archivo .onnx que luego debes copiar a:")
    print("app/src/main/assets/models/base/tts_model_es_cl.onnx")
    print("====================================================================\n")

if __name__ == "__main__":
    main()
