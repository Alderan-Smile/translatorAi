import os
from faster_whisper import WhisperModel
import tempfile
import pyaudio
import threading
import numpy as np
import soundfile as sf
import time
import ffmpeg
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

    def transcribir_archivo_a_srt(self, ruta_entrada, modelo_audio):
        """
        Transcribe un archivo de audio/video a subtítulos SRT usando faster-whisper.
        Extrae el audio a WAV temporalmente para saltar errores/cortes en el stream.
        """
        base, _ = os.path.splitext(os.path.basename(ruta_entrada))
        ruta_salida_srt = f"./subtitle/{base}.srt"

        # Extraer audio a WAV temporal
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name
        try:
            # -err_detect ignore_err: ignora errores de stream
            ffmpeg.input(ruta_entrada).output(
                wav_path, 
                format='wav', 
                acodec='pcm_s16le', 
                ac=1, 
                ar='16000', 
                loglevel='error', 
                **{'err_detect': 'ignore_err'}
            ).overwrite_output().run()
        except Exception as e:
            print(f"Error extrayendo audio: {e}")
            return

        # Obtener duración total del WAV
        try:
            probe = ffmpeg.probe(wav_path)
            duration = float(probe['format']['duration'])
        except Exception as e:
            print(f"No se pudo obtener la duración del audio extraído: {e}")
            duration = None

        if duration:
            minutos = int(duration // 60)
            segundos = int(duration % 60)
            print(f"Duración total del audio extraído: {minutos} min {segundos} s ({duration:.2f} segundos)")
        else:
            print("Duración total del audio extraído: desconocida")

        local_model_path = os.path.abspath(f"../resources/faster-whisper-{modelo_audio}-int8")
        model = WhisperModel(
            local_model_path, 
            device="cuda",
            ##device="cpu",
            compute_type="int8"
        )

        print(f"Transcribiendo archivo: {ruta_entrada}")
        segments, info = model.transcribe(wav_path, beam_size=5, language="es", task="transcribe")

        def format_timestamp(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int((seconds - int(seconds)) * 1000)
            return f"{h:02}:{m:02}:{s:02},{ms:03}"

        with open(ruta_salida_srt, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, 1):
                start = format_timestamp(segment.start)
                end = format_timestamp(segment.end)
                text = segment.text.strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
                if duration:
                    porcentaje = min(100, (segment.end / duration) * 100)
                    print(
                        f"[SRT] Progreso: {porcentaje:6.2f}% | Tiempo: {segment.end:.1f}s / {duration:.1f}s",
                        end='\r'
                    )
        print("\nSRT generado en:", ruta_salida_srt)

        # Limpia el archivo temporal
        try:
            os.remove(wav_path)
        except Exception:
            pass

    def clear_console(self):
        print('\r' + ' ' * 120 + '\r', end='')

    def decodeString(self, string):
        try:
            return string.encode('latin1').decode('utf-8')
        except UnicodeDecodeError:
            return string.encode('cp1252').decode('utf-8', errors='ignore')
