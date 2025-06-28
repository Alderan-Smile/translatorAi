from service.translator import translatorFull
from service.spToText import speechToText
import pyaudio
import sys
if sys.platform == "win32":
    import os
    os.system("chcp 65001")

translat = translatorFull()
talkToText = speechToText()
p = pyaudio.PyAudio()

#inputText = "Bienvenido invocador, ¿cómo estás?, soy un bot de traducción, ¿en qué puedo ayudarte hoy?"
#translat.realSub(inputText)
print("Bienvenido al subtitulador automatico.\n")
print("Dispositivos de entrada de audio disponibles:")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
        print(f"{i}: {info['name']}")

device_index = int(input("Selecciona el índice del audio a usar: "))

print("Selecciona el modelo de Whisper:")
print("1. base")
print("2. small")
print("3. medium")
print("4. large")
model_choice = input("Ingresa el número del modelo: ").strip()
if model_choice == "1":
    WHISPER_MODEL = "base"
elif model_choice == "2":
    WHISPER_MODEL = "small"
elif model_choice == "3":
    WHISPER_MODEL = "medium"
elif model_choice == "4":
    WHISPER_MODEL = "large"
else:
    print("Modelo no válido, usando 'medium' por defecto.")
    WHISPER_MODEL = "medium"

talkToText.escuchar(device_index, WHISPER_MODEL)