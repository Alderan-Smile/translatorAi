from service.spToText import speechToText
import os
import sys

def main():
    print("Subtitulador de archivos (audio/video) a SRT\n")

    # Buscar modelos Whisper disponibles
    resources_path = os.path.abspath("../resources/")
    modelos_whisper = []
    for d in os.listdir(resources_path):
        if d.startswith("faster-whisper-") and os.path.isdir(os.path.join(resources_path, d)):
            nombre = d.replace("faster-whisper-", "").replace("-int8", "").replace("-int4", "")
            if nombre not in modelos_whisper:
                modelos_whisper.append(nombre)

    if not modelos_whisper:
        print("No se encontraron modelos Whisper en la carpeta resources.")
        sys.exit(1)

    print("Modelos de Whisper disponibles:")
    for idx, nombre in enumerate(modelos_whisper, 1):
        print(f"{idx}: {nombre}")

    opcion_whisper = input("Selecciona el modelo de Whisper a usar: ").strip()
    try:
        modelo_audio = modelos_whisper[int(opcion_whisper) - 1]
    except (ValueError, IndexError):
        print("Selección inválida, usando el primero por defecto.")
        modelo_audio = modelos_whisper[0]

    ruta_entrada = input("Ruta del archivo de audio/video a transcribir: ").strip()
    if not os.path.isfile(ruta_entrada):
        print("Archivo no encontrado.")
        sys.exit(1)

    stt = speechToText()
    stt.transcribir_archivo_a_srt(ruta_entrada, modelo_audio)

if __name__ == "__main__":
    main()