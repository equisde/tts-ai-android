@echo off
setlocal enabledelayedexpansion
title F5-TTS AI Server - Chilean Edition

echo =======================================================
echo    Iniciador Inteligente de Servidor TTS IA
echo =======================================================

:: 1. Verificar Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no esta instalado. Por favor instala Python 3.10+
    pause
    exit /b
)

:: 2. Crear Entorno Virtual (VENV) para evitar conflictos
if not exist "venv" (
    echo [INFO] Creando entorno virtual de Python...
    python -m venv venv
)

:: 3. Activar VENV
call venv\Scripts\activate

:: 4. Verificar e instalar requerimientos
echo [INFO] Verificando dependencias...
pip install -r requirements.txt

:: 5. Verificar modelos
if not exist "models\base\F5_Transformer.onnx" (
    echo [ALERTA] No se detectaron los modelos ONNX en server/models/base/
    echo [ALERTA] Por favor, copia los archivos .onnx y vocab.txt antes de continuar.
    pause
)

:: 6. Lanzar Servidor con prioridad de CPU alta
echo [INFO] Lanzando servidor FastAPI a maxima potencia...
echo [INFO] El servidor estara disponible en http://localhost:8000
echo [INFO] Presiona CTRL+C para detener.
echo -------------------------------------------------------

python main.py

pause