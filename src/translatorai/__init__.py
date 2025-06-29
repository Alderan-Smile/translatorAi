from service.translator import translatorFull
from service.spToText import speechToText
import pyaudio
import sys
if sys.platform == "win32":
    import os
    os.system("chcp 65001")
    sys.stdout.reconfigure(encoding='utf-8')

translat = translatorFull()
talkToText = speechToText()
p = pyaudio.PyAudio()

#inputText = "Bienvenido invocador, ¿cómo estás?, soy un bot de traducción, ¿en qué puedo ayudarte hoy?"
#translat.realSub(inputText)
print("Bienvenido al subtitulador automatico.\n")

# Mostrar dispositivos de entrada habilitados
print("Dispositivos de entrada de audio disponibles:")
input_devices = []
default_host_api_index = p.get_default_host_api_info()["index"]
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0 and info["hostApi"] == default_host_api_index:
        input_devices.append((i, info["name"]))
        print(f"{i}: {info['name']}")

if not input_devices:
    print("No hay dispositivos de entrada de audio habilitados.")
    sys.exit(1)

# Mostrar dispositivos de salida habilitados
print("\nDispositivos de salida de audio disponibles:")
output_devices = []
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxOutputChannels"] > 0 and info["hostApi"] == default_host_api_index:
        output_devices.append((i, info["name"]))
        print(f"{i}: {info['name']}")

if not output_devices:
    print("No hay dispositivos de salida de audio habilitados.")

device_index = int(input("\nSelecciona el índice del audio de entrada a usar: "))
talkToText.clear_console()

# Menú de idiomas
LANGUAGES = {
    "1": ("es", "Español"),
    "2": ("en", "Inglés"),
    "3": ("fr", "Francés"),
    "4": ("de", "Alemán"),
    "5": ("it", "Italiano"),
    "6": ("pt", "Portugués"),
    "7": ("ru", "Ruso"),
    "8": ("zh", "Chino"),
    "9": ("ja", "Japonés"),
    "10": ("ko", "Coreano"),
}

print("\nSelecciona el idioma de entrada:")
for k, v in LANGUAGES.items():
    print(f"{k}. {v[1]}")
input_lang_choice = input("Número de idioma de entrada: ").strip()
input_lang = LANGUAGES.get(input_lang_choice, ("es",))[0]

TIPO_SALIDA = {
    "1": "transcribe",
    "2": "translate"
}

print("\nSelecciona el tipo de salida:")
for k, v in TIPO_SALIDA.items():
    print(f"{k}. {v.capitalize()}")
output_choice = input("Número de tipo de salida: ").strip()
type = TIPO_SALIDA.get(output_choice, "transcribe")

talkToText.clear_console()


# Mostrar modelos de Whisper disponibles
print("Selecciona el modelo de Whisper:")
print("1. base")
print("2. small")
print("3. medium")
print("4. large")
print("5. turbo")
model_choice = input("Ingresa el número del modelo: ").strip()
if model_choice == "1":
    WHISPER_MODEL = "base"
elif model_choice == "2":
    WHISPER_MODEL = "small"
elif model_choice == "3":
    WHISPER_MODEL = "medium"
elif model_choice == "4":
    WHISPER_MODEL = "large"
elif model_choice == "5":
    WHISPER_MODEL = "turbo"
else:
    print("Modelo no válido, usando 'medium' por defecto.")
    WHISPER_MODEL = "medium"

talkToText.clear_console()

talkToText.escuchar(device_index, WHISPER_MODEL, input_lang,type)