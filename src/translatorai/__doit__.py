from service.spToText import speechToText
import os
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "resources"))


def main():
    print("Subtitulador de archivos (audio/video) a SRT\n")

    # Buscar modelos Whisper disponibles
    resources_path = RESOURCES_DIR
    modelos_whisper = []
    for d in os.listdir(resources_path):
        if (d.startswith("faster-whisper-") or d.startswith("faster-distil-whisper-")) and os.path.isdir(os.path.join(resources_path, d)):
            nombre = d.replace("faster-whisper-", "").replace("faster-distil-whisper-", "").replace("-int8", "-i8").replace("-int4", "-i4").replace("-float16", "-f16")
            if nombre not in modelos_whisper:
                modelos_whisper.append(nombre)
    modelos_whisper.sort()

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

    permitir_fallback = True
    base_modelo = modelo_audio
    for sufijo in ("-i8", "-i4", "-f16"):
        if base_modelo.endswith(sufijo):
            base_modelo = base_modelo[:-len(sufijo)]
            break

    if base_modelo in {"large-v3", "large-v3-turbo"}:
        modo_directo = input(
            "Seleccionaste un modelo v3. ¿Intentar ejecución directa sin fallback automático? (s/N): "
        ).strip().lower()
        if modo_directo in {"s", "si", "sí", "y", "yes"}:
            permitir_fallback = False

    ruta_entrada = input("Ruta del archivo de audio/video a transcribir: ").strip()
    if not os.path.isfile(ruta_entrada):
        print("Archivo no encontrado.")
        sys.exit(1)

    stt = speechToText()
    try:
        stt.transcribir_archivo_a_srt(ruta_entrada, modelo_audio, permitir_fallback=permitir_fallback)
    except RuntimeError as e:
        msg = str(e)
        if "128/80 mel bins" in msg and not permitir_fallback:
            print(f"\n{msg}")
            reintentar = input("¿Reintentar con fallback automático? (S/n): ").strip().lower()
            if reintentar not in {"n", "no"}:
                stt.transcribir_archivo_a_srt(ruta_entrada, modelo_audio, permitir_fallback=True)
                return
        raise

if __name__ == "__main__":
    main()