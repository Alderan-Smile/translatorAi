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
import itertools
from service.fileWriter import controlArchivos
import datetime

if sys.platform == "win32":
    import os
    os.system("chcp 65001")
    sys.stdout.reconfigure(encoding='utf-8')


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
RESOURCES_DIR = os.path.abspath(os.path.join(APP_DIR, "..", "resources"))
SUBTITLE_DIR = os.path.join(APP_DIR, "subtitle")

class speechToText:

    def _nombre_visible_modelo(self, directory_name):
        return (
            directory_name
            .replace("faster-whisper-", "")
            .replace("faster-distil-whisper-", "")
            .replace("-int8", "-i8")
            .replace("-int4", "-i4")
            .replace("-float16", "-f16")
        )

    def _base_modelo(self, nombre_modelo):
        for sufijo in ("-i8", "-i4", "-f16"):
            if nombre_modelo.endswith(sufijo):
                return nombre_modelo[:-len(sufijo)]
        return nombre_modelo

    def _resolver_iterador_segmentos(self, segments):
        """Fuerza la evaluación inicial del generador para capturar errores tempranos de forma controlada."""
        iterator = iter(segments)
        first = next(iterator, None)
        if first is None:
            return iter(())
        return itertools.chain([first], iterator)

    def _es_error_mel_bins(self, error):
        msg = str(error)
        return "shape (1, 128, 3000)" in msg and "shape (1, 80, 3000)" in msg

    def _texto_muy_repetitivo(self, text):
        limpio = " ".join(text.lower().split())
        if not limpio:
            return True

        tokens = limpio.split(" ")
        if len(tokens) < 6:
            return False

        # Marca bucles largos con muy poca variedad de palabras.
        variedad = len(set(tokens)) / len(tokens)
        return variedad < 0.25

    def _es_segmento_hallucinado(self, segment):
        text = segment.text.strip()
        if not text:
            return True

        duracion = max(0.0, float(segment.end) - float(segment.start))
        avg_logprob = getattr(segment, "avg_logprob", None)
        no_speech_prob = getattr(segment, "no_speech_prob", None)
        compression_ratio = getattr(segment, "compression_ratio", None)

        if no_speech_prob is not None and no_speech_prob > 0.92 and duracion < 1.0:
            return True
        if avg_logprob is not None and avg_logprob < -2.2 and duracion < 1.2:
            return True
        if compression_ratio is not None and compression_ratio > 3.7 and len(text) > 20:
            return True
        if self._texto_muy_repetitivo(text) and duracion > 1.2:
            return True

        return False

    def _listar_modelos_locales(self):
        modelos = []
        for d in os.listdir(RESOURCES_DIR):
            full = os.path.join(RESOURCES_DIR, d)
            if (d.startswith("faster-whisper-") or d.startswith("faster-distil-whisper-")) and os.path.isdir(full):
                nombre = self._nombre_visible_modelo(d)
                if nombre not in modelos:
                    modelos.append(nombre)
        return modelos

    def _modelo_fallback(self, modelo_actual):
        modelos = self._listar_modelos_locales()
        base_actual = self._base_modelo(modelo_actual)
        sufijo_actual = ""
        for sufijo in ("-i8", "-i4", "-f16"):
            if modelo_actual.endswith(sufijo):
                sufijo_actual = sufijo
                break

        prioridad = ["large-v2", "large", "medium", "small", "base"]
        for candidato in prioridad:
            if candidato == base_actual:
                continue

            # Prefer same quantization/precision suffix when available.
            if sufijo_actual:
                candidato_con_sufijo = f"{candidato}{sufijo_actual}"
                if candidato_con_sufijo in modelos:
                    return candidato_con_sufijo

            for disponible in modelos:
                if self._base_modelo(disponible) == candidato:
                    return disponible
        return None

    def _crear_modelo(self, modelo_audio):
        base = modelo_audio
        sufijo = ""
        compute_type = "auto"

        for s, ct in (("-i8", "int8"), ("-i4", "int8_float16"), ("-f16", "float16")):
            if modelo_audio.endswith(s):
                base = modelo_audio[:-len(s)]
                sufijo = s
                compute_type = ct
                break

        if not sufijo:
            # Backward compatibility with old naming (without explicit suffix).
            sufijo = "-i8"
            compute_type = "int8"

        suffix_dir = {"-i8": "-int8", "-i4": "-int4", "-f16": "-float16"}[sufijo]
        candidatos = [
            os.path.join(RESOURCES_DIR, f"faster-whisper-{base}{suffix_dir}"),
            os.path.join(RESOURCES_DIR, f"faster-distil-whisper-{base}{suffix_dir}"),
        ]

        for local_model_path in candidatos:
            if os.path.isdir(local_model_path):
                return WhisperModel(local_model_path, device="cuda", compute_type=compute_type)

        raise FileNotFoundError(
            f"No se encontró carpeta de modelo para '{modelo_audio}'. Se esperaba una de: {candidatos}"
        )

    def escuchar(self, device_index, name_device, modelo_audio,channels_device, permitir_fallback=True):
        
        print(f"Modelo Whiper Cargado: {modelo_audio}")
        print(f"Canales usados por {name_device}: {channels_device}")
        model = self._crear_modelo(modelo_audio)
        
        file_ctrl = controlArchivos()

        FORMAT = pyaudio.paInt16
        CHANNELS = channels_device
        RATE = 16000
        CHUNK_SIZE_MS = 2000
        CHUNK = int(RATE * CHUNK_SIZE_MS / 1000)
        fecha = datetime.datetime.now().strftime("%Y-%m-%d")
        hora = datetime.datetime.now().strftime("%H:%M:%S")
        TXT_FILE = os.path.join(SUBTITLE_DIR, "subt.txt")
        #TXT_FILE2 = f"./subtitle/backup_{fecha}.txt"
        #TXT_FILE3 = f"./subtitle/Segmento_{fecha}.txt"

        file_ctrl.compruebaCarpetas(SUBTITLE_DIR)
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
                try:
                    segments, info = model.transcribe(audio_np, beam_size=5, language="es", task="translate")
                    segments = self._resolver_iterador_segmentos(segments)
                except ValueError as e:
                    if self._es_error_mel_bins(e):
                        if not permitir_fallback:
                            raise RuntimeError(
                                f"El modelo '{modelo_audio}' falló por incompatibilidad 128/80 mel bins y el fallback está desactivado."
                            ) from e

                        fallback = self._modelo_fallback(modelo_audio)
                        if not fallback:
                            raise RuntimeError(
                                "El modelo seleccionado requiere 128 mel bins y no hay un modelo local compatible para fallback. "
                                "Actualiza faster-whisper/ctranslate2 o instala un modelo Whisper alternativo (por ejemplo large-v2)."
                            ) from e

                        print(
                            f"\nAviso: el modelo '{modelo_audio}' no es compatible con la extracción actual (128/80 mel bins). "
                            f"Cambiando automáticamente a '{fallback}'."
                        )
                        modelo_audio = fallback
                        model = self._crear_modelo(modelo_audio)
                        segments, info = model.transcribe(audio_np, beam_size=5, language="es", task="translate")
                        segments = self._resolver_iterador_segmentos(segments)
                    else:
                        raise
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

    def transcribir_archivo_a_srt(self, ruta_entrada, modelo_audio, permitir_fallback=True):
        """
        Transcribe un archivo de audio/video a subtítulos SRT usando faster-whisper.
        Extrae el audio a WAV temporalmente para saltar errores/cortes en el stream.
        """
        base, _ = os.path.splitext(os.path.basename(ruta_entrada))
        ruta_salida_srt = os.path.join(SUBTITLE_DIR, f"{base}.srt")

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

        model = self._crear_modelo(modelo_audio)

        TRANSCRIBE_OPTS = dict(
            beam_size=5,
            language="es",
            task="transcribe",
            # Anti-hallucination settings (permisivos para no perder diálogos cortos/repetitivos)
            condition_on_previous_text=False,   # evita propagar texto alucinado entre chunks
            vad_filter=True,                     # reactivado para reducir alucinaciones en silencios
            vad_parameters=dict(
                threshold=0.22,                 # capta voz baja sin abrir demasiado ruido
                min_speech_duration_ms=90,      # mantiene diálogos ultracortos
                min_silence_duration_ms=450,    # equilibrio entre cortes y continuidad
                speech_pad_ms=350,              # añade contexto útil alrededor de voz
            ),
            log_prob_threshold=-2.2,            # permite segmentos difíciles, pero no extremos
            no_speech_threshold=0.9,            # solo descarta si hay >90% prob de NO-habla
            compression_ratio_threshold=4.0,    # evita filtrar repeticiones naturales cortas
        )

        print(f"Transcribiendo archivo: {ruta_entrada}")
        try:
            segments, info = model.transcribe(wav_path, **TRANSCRIBE_OPTS)
            segments = self._resolver_iterador_segmentos(segments)
        except ValueError as e:
            if not self._es_error_mel_bins(e):
                raise

            if not permitir_fallback:
                raise RuntimeError(
                    f"El modelo '{modelo_audio}' falló por incompatibilidad 128/80 mel bins y el fallback está desactivado."
                ) from e

            fallback = self._modelo_fallback(modelo_audio)
            if not fallback:
                raise RuntimeError(
                    "Error de incompatibilidad 128/80 mel bins y no se encontró un modelo local compatible para fallback. "
                    "Actualiza faster-whisper/ctranslate2 o instala un modelo Whisper alternativo (por ejemplo large-v2)."
                ) from e

            print(
                f"Aviso: el modelo '{modelo_audio}' no es compatible con la extracción actual (128/80 mel bins). "
                f"Reintentando con '{fallback}'."
            )
            modelo_audio = fallback
            model = self._crear_modelo(modelo_audio)
            segments, info = model.transcribe(wav_path, **TRANSCRIBE_OPTS)
            segments = self._resolver_iterador_segmentos(segments)

        def format_timestamp(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int((seconds - int(seconds)) * 1000)
            return f"{h:02}:{m:02}:{s:02},{ms:03}"

        with open(ruta_salida_srt, "w", encoding="utf-8") as f:
            i = 1
            descartados = 0
            for segment in segments:
                if self._es_segmento_hallucinado(segment):
                    descartados += 1
                    continue

                start = format_timestamp(segment.start)
                end = format_timestamp(segment.end)
                text = segment.text.strip()
                if not text:
                    continue

                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
                i += 1
                if duration:
                    porcentaje = min(100, (segment.end / duration) * 100)
                    print(
                        f"[SRT] Progreso: {porcentaje:6.2f}% | Tiempo: {segment.end:.1f}s / {duration:.1f}s",
                        end='\r'
                    )
        if descartados:
            print(f"\nSegmentos descartados por baja confianza/repetición: {descartados}")
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
