import os
import whisper
import pyaudio
import numpy as np
import soundfile as sf
import time
import sys
import gc

class speechToText:

    def escuchar(self, device_index, WHISPER_MODEL, input_lang="es",type="transcribe"):
        
        #WHISPER_MODEL = "medium"  # Puedes cambiar a "base", "small", "medium", "large" según tus necesidades

        model = whisper.load_model(WHISPER_MODEL)
        print(f"Modelo cargado: {WHISPER_MODEL}")

        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK_SIZE_MS = 3000
        CHUNK = int(RATE * CHUNK_SIZE_MS / 1000)
        TXT_FILE = "./subtitle/subt.txt"

        os.makedirs(os.path.dirname("./subtitle/"), exist_ok=True)
        with open(TXT_FILE, "a", encoding="utf-8"):
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
        try:
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                audio_float32 = audio_data.astype(np.float32) / 32768.0
                start_time = time.time()
                result = model.transcribe(
                    audio_float32, 
                    fp16=False, 
                    language=input_lang, 
                    task="transcribe" if type != "transcribe" else "translate"
                )
                transcribed_text = result["text"].strip()
                end_time = time.time()
                energy = np.linalg.norm(audio_float32)
                frases_ignoradas = [
                    "thanks for watching!<br>\n",
                    "gracias por ver!<br>\n",
                    "thanks for watching!<br>",
                    "thanks for watching!",
                    "thanks for watching",
                    "bye!<br>\n",
                    "bye!<br>",
                    "bye!",
                    "bye",
                    "done.<br>\n",
                    "done.<br>",
                    "done.",
                    "done",
                    "see you later!<br>\n",
                    "see you later!<br>",
                    "see you later!",
                    "see you later",
                    "thanks you!",
                    "thanks you for watching!<br>\n",
                    "thanks you for watching!<br>",
                    "thanks you for watching!",
                    "thanks you for watching",
                ]
                if transcribed_text and energy > 0.09 and transcribed_text.lower() not in frases_ignoradas:
                    #print(f"[Latencia: {end_time - start_time:.2f}s] Tú: {transcribed_text}")
                    with open(TXT_FILE, "a", encoding="utf-8") as f:
                        f.write(f"{transcribed_text}<br>\n")
                
                if time.time() - last_cleanup > 3:
                    if os.path.exists(TXT_FILE):
                        with open(TXT_FILE, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                        if len(lines) > 1:
                            with open(TXT_FILE, "w", encoding="utf-8") as f:
                                f.writelines(lines[1:])
                    last_cleanup = time.time()
                
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
        if sys.platform == "win32":
            os.system("cls")
        else:
            os.system("clear")
