import os
from faster_whisper import WhisperModel
import pyaudio
import threading
import numpy as np
import soundfile as sf
import time
import sys
import gc
from service.fileWriter import controlArchivos
import datetime
if sys.platform == "win32":
    import os
    os.system("chcp 65001")
    sys.stdout.reconfigure(encoding='utf-8')

class speechToText:

    def escuchar(self, device_index, name_device, modelo_audio,channels_device):
        
        print(f"Modelo Whiper Cargado: {modelo_audio}")
        print(f"Canales usados por {name_device}: {channels_device}")
        local_model_path = os.path.abspath(f"../resources/faster-whisper-{modelo_audio}-int8")
        model = WhisperModel(local_model_path, device="cuda", compute_type="int8")
        
        file_ctrl = controlArchivos()

        FORMAT = pyaudio.paInt16
        CHANNELS = channels_device
        RATE = 16000
        CHUNK_SIZE_MS = 2000
        CHUNK = int(RATE * CHUNK_SIZE_MS / 1000)
        fecha = datetime.datetime.now().strftime("%Y-%m-%d")
        hora = datetime.datetime.now().strftime("%H:%M:%S")
        TXT_FILE = "./subtitle/subt.txt"
        #TXT_FILE2 = f"./subtitle/backup_{fecha}.txt"
        #TXT_FILE3 = f"./subtitle/Segmento_{fecha}.txt"

        file_ctrl.compruebaCarpetas("./subtitle/")
        file_ctrl.compruebaArchivos(TXT_FILE)
        #file_ctrl.compruebaArchivos(TXT_FILE2)
        #file_ctrl.compruebaArchivos(TXT_FILE3)

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, 
                        channels=CHANNELS, 
                        rate=RATE, 
                        input=True, 
                        input_device_index=device_index,
                        frames_per_buffer=CHUNK)
        print("Grabando audio...")
        frases_ignoradas = [
            "Thank you for watching!",
            "Thanks for watching!",
        ]

        def limpiar_subtitulos_periodicamente():
            while True:
                ahora = time.time()
                cola_subtitulos[:] = [(txt, t) for txt, t in cola_subtitulos if ahora - t < 3]
                file_ctrl.borraArchivos(TXT_FILE, cola_subtitulos)
                time.sleep(0.5)  # Ajusta el intervalo según necesidad

        cola_subtitulos = []
        limpiar_thread = threading.Thread(target=limpiar_subtitulos_periodicamente, daemon=True)
        limpiar_thread.start()

        try:
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                start_time = time.time()
                segments, info = model.transcribe(audio_np, beam_size=5, language="es", task="translate")
                for segment in segments:
                    segmentodsTrip = segment.text.strip()
                    if segmentodsTrip not in frases_ignoradas:
                        end_time = time.time()
                        energy = np.linalg.norm(audio_np)
                        latencia = end_time - start_time
                        print(f"[{name_device}] Latencia: {latencia:.2f}s | Energía: {energy:.3f} ", end='\r')
                        if energy > 0.200:
                            timestamp = time.time()
                            cola_subtitulos.append((segmentodsTrip, timestamp))
                            file_ctrl.escribeArchivos(TXT_FILE, f"{segmentodsTrip}<br>\n")
                
                del audio_np
                gc.collect()
                
        except KeyboardInterrupt:
            print("\nGrabación detenida.")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Stream cerrado y PyAudio terminado.")

    def clear_console(self):
        print('\r' + ' ' * 120 + '\r', end='')

    def decodeString(self, string):
        try:
            return string.encode('latin1').decode('utf-8')
        except UnicodeDecodeError:
            return string.encode('cp1252').decode('utf-8', errors='ignore')
