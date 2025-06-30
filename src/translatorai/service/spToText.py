import threading
import queue
import os
from faster_whisper import WhisperModel
import pyaudio
import numpy as np
import soundfile as sf
import time
import sys
import gc
import datetime
from service.translator import translatorFull
if sys.platform == "win32":
    import os
    os.system("chcp 65001")
    sys.stdout.reconfigure(encoding='utf-8')

class speechToText:

    def escuchar(self, device_index, name_device, modelo_audio, modelo_traduccion):

        translat = translatorFull()
        
        # Ruta local al modelo Faster-Whisper large-v2 INT8
        local_model_path = os.path.abspath(f"../resources/faster-whisper-{modelo_audio}-int8")
        model = WhisperModel(local_model_path, device="cuda", compute_type="int8")
        print(f"Modelo Faster-Whisper cargado desde: {local_model_path}")

        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK_SIZE_MS = 3000
        CHUNK = int(RATE * CHUNK_SIZE_MS / 1000)
        fecha_hora = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        TXT_FILE = "./subtitle/subt.txt"
        TXT_FILE2 = f"./subtitle/backup_{fecha_hora}.txt"
        TXT_FILE3 = f"./subtitle/backuptrad_{fecha_hora}.txt"

        os.makedirs(os.path.dirname("./src/translatorai/subtitle/"), exist_ok=True)
        with open(TXT_FILE2, "a", encoding="utf-8"):
            pass

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, 
                        channels=CHANNELS, 
                        rate=RATE, 
                        input=True, 
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)
        print("Grabando audio...")

        # --- Cambios para paralelizar ---
        text_queue = queue.Queue()

        def transcribir_audio():
            try:
                while True:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    audio_float32 = audio_data.astype(np.float32) / 32768.0
                    start_time = time.time()
                    segments, info = model.transcribe(audio_float32, language="es", task="transcribe")
                    transcribed_text = ""
                    for segment in segments:
                        transcribed_text += segment.text.strip() + " "
                    transcribed_text = transcribed_text.strip()
                    end_time = time.time()
                    energy = np.linalg.norm(audio_float32)
                    latencia = end_time - start_time
                    print(f"[{name_device}] Latencia: {latencia:.2f}s | Energía: {energy:.3f}   ", end='\r')
                    frases_ignoradas = [
                        "¡Gracias por ver el vídeo!",
                        "¡Adiós!",
                        "¡Gracias por ver!",
                        "Subtítulos realizados por la comunidad de Amara.org",
                        "Gracias por ver",
                        "y nos vemos en el próximo video",
                        "¡Gracias!"
                    ]
                    if transcribed_text and energy > 0.200 and transcribed_text not in frases_ignoradas:
                        # Enviar a la cola para traducir
                        text_queue.put((transcribed_text, TXT_FILE, TXT_FILE3, TXT_FILE2, latencia, energy))
                    del audio_data
                    del audio_float32
                    gc.collect()
            except KeyboardInterrupt:
                print("\nGrabación detenida.")
            finally:
                # Señal de parada para el hilo de traducción
                text_queue.put(None)

        def traducir_texto():
            while True:
                item = text_queue.get()
                if item is None:
                    break
                transcribed_text, TXT_FILE, TXT_FILE3, TXT_FILE2, latencia, energy = item
                if modelo_traduccion == "INT8":
                    translat.subI8(transcribed_text, TXT_FILE, TXT_FILE3)
                elif modelo_traduccion == "INT4":
                    translat.subI4(transcribed_text, TXT_FILE, TXT_FILE3)
                else:
                    translat.subOr(transcribed_text, TXT_FILE, TXT_FILE3)
                with open(TXT_FILE2, "a", encoding="utf-8") as f2:
                    f2.write(f" ({name_device}) [Latencia: {latencia:.2f}s] | [Energía: {energy:.3f}] Tu: {transcribed_text}\n")
                translat.comprobadorSubtitulos(TXT_FILE)

        hilo_transcripcion = threading.Thread(target=transcribir_audio)
        hilo_traduccion = threading.Thread(target=traducir_texto)
        hilo_transcripcion.start()
        hilo_traduccion.start()

        try:
            hilo_transcripcion.join()
        except KeyboardInterrupt:
            print("\nGrabación detenida.")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            hilo_traduccion.join()
            print("Stream cerrado y PyAudio terminado.")

    def clear_console(self):
        print('\r' + ' ' * 120 + '\r', end='')

    def decodeString(self, string):
        try:
            return string.encode('latin1').decode('utf-8')
        except UnicodeDecodeError:
            return string.encode('cp1252').decode('utf-8', errors='ignore')