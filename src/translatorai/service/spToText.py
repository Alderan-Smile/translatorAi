import os
import whisper
import pyaudio
import numpy as np
import soundfile as sf
import time
import sys
import gc
import datetime
if sys.platform == "win32":
    import os
    os.system("chcp 65001")
    sys.stdout.reconfigure(encoding='utf-8')

class speechToText:

    def escuchar(self, device_index, WHISPER_MODEL,name_device):
        
        #WHISPER_MODEL = "medium"  # Puedes cambiar a "base", "small", "medium", "large" según tus necesidades

        model = whisper.load_model(WHISPER_MODEL)
        print(f"Modelo cargado: {WHISPER_MODEL}")

        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK_SIZE_MS = 3000
        CHUNK = int(RATE * CHUNK_SIZE_MS / 1000)
        fecha_hora = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        TXT_FILE = "./subtitle/subt.txt"
        TXT_FILE2 = f"./subtitle/backup_{fecha_hora}.txt"

        os.makedirs(os.path.dirname("./subtitle/"), exist_ok=True)
        with open(TXT_FILE, "a", encoding="utf-8"):
            pass
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

        last_cleanup = time.time()
        cola_subtitulos = []

        try:
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                audio_float32 = audio_data.astype(np.float32) / 32768.0
                start_time = time.time()
                result = model.transcribe(
                    audio_float32, 
                    fp16=False, 
                    language="es", 
                    task="translate")
                transcribed_text = result["text"].strip()
                end_time = time.time()
                energy = np.linalg.norm(audio_float32)
                latencia = end_time - start_time
                print(f"[{name_device}] Latencia: {latencia:.2f}s | Energía: {energy:.3f}   ", end='\r')
                frases_ignoradas = [
                    "Thank you for watching!",
                    "Thanks for watching!"
                ]
                if transcribed_text and energy > 0.200 and transcribed_text not in frases_ignoradas:
                    timestamp = time.time()
                    cola_subtitulos.append((transcribed_text, timestamp))
                    with open(TXT_FILE, "a", encoding="utf-8") as f:
                        f.write(f"{transcribed_text}<br>\n")
                    with open(TXT_FILE2, "a", encoding="utf-8") as f2:
                        f2.write(f" ({name_device}) [Latencia: {latencia:.2f}s] | [Energía: {energy:.3f}] Tu: {transcribed_text}\n")
                
                ahora = time.time()
                cola_subtitulos = [(txt, t) for txt, t in cola_subtitulos if ahora - t < 2]
                with open(TXT_FILE, "w", encoding="utf-8") as f:
                    for txt, _ in cola_subtitulos:
                        f.write(f"{txt}<br>\n")
                
                del audio_data
                del audio_float32
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
