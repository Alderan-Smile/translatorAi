from service.spToText import speechToText
import pyaudio
import sys
if sys.platform == "win32":
    import os
    os.system("chcp 65001")
    sys.stdout.reconfigure(encoding='utf-8')

talkToText = speechToText()
p = pyaudio.PyAudio()
nameDevice: str = ""

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
        deviceName = talkToText.decodeString(info["name"])
        input_devices.append((i, deviceName))
        print(f"{i}: {deviceName}")

if not input_devices:
    print("No hay dispositivos de entrada de audio habilitados.")
    sys.exit(1)

# Mostrar dispositivos de salida habilitados
print("\nDispositivos de salida de audio disponibles:")
output_devices = []
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxOutputChannels"] > 0 and info["hostApi"] == default_host_api_index:
        deviceName = talkToText.decodeString(info["name"])
        output_devices.append((i, deviceName))
        print(f"{i}: {deviceName}")

if not output_devices:
    print("No hay dispositivos de salida de audio habilitados.")

device_index = int(input("\nSelecciona el índice del audio de entrada a usar: "))
nameDevice = talkToText.decodeString(p.get_device_info_by_index(device_index)['name'])
for _ in range(14):
    print('\033[F\033[K', end='')  # Limpiar líneas anteriores


# Mostrar modelos de Whisper disponibles
print("Selecciona el modelo de Whisper:")
print("1. base")
print("2. small")
print("3. medium")
print("4. large")
print("5. turbo")
print("6. large-v2")
print("7. large-v3")
model_choice = input("Ingresa el número del modelo: ").strip()
for _ in range(9):
    print('\033[F\033[K', end='')
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
elif model_choice == "6":
    WHISPER_MODEL = "large-v2"
elif model_choice == "7":
    WHISPER_MODEL = "large-v3"
else:
    print("Modelo no válido, usando 'medium' por defecto.")
    WHISPER_MODEL = "medium"


talkToText.escuchar(device_index, WHISPER_MODEL, nameDevice)