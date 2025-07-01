from service.spToText import speechToText
import os
import pyaudio
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'resources')))
from optimumPrime import optimizadorModelos
if sys.platform == "win32":
    os.system("chcp 65001")
    sys.stdout.reconfigure(encoding='utf-8')

talkToText = speechToText()
optiModel = optimizadorModelos()
p = pyaudio.PyAudio()
nameDevice: str = ""

print("Bienvenido al subtitulador automatico.\n")

# Mostrar dispositivos de entrada habilitados
rongoLineasDispo = 5
print("Dispositivos de entrada de audio disponibles:")
input_devices = []
default_host_api_index = p.get_default_host_api_info()["index"]
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0 and info["hostApi"] == default_host_api_index:
        deviceName = talkToText.decodeString(info["name"])
        input_devices.append((i, deviceName))
        print(f"{i}: {deviceName}")
        rongoLineasDispo+=1

if not input_devices:
    print("No hay dispositivos de entrada de audio habilitados.")
    rongoLineasDispo+=1
    sys.exit(1)

device_index = int(input("\nSelecciona el índice del audio de entrada a usar: "))
nameDevice = talkToText.decodeString(p.get_device_info_by_index(device_index)['name'])
infor = p.get_device_info_by_index(device_index)
channels_device = infor["maxInputChannels"]

for _ in range(rongoLineasDispo):
    print('\033[F\033[K', end='')  # Limpiar líneas anteriores

# Buscar modelos Whisper disponibles
rongoLineaswhisper = 2
resources_path = os.path.abspath("../resources/")
modelos_whisper = []
for d in os.listdir(resources_path):
    if d.startswith("faster-whisper-") and os.path.isdir(os.path.join(resources_path, d)):
        # Extraer nombre de modelo (medium, large, large-v2, etc.)
        nombre = d.replace("faster-whisper-", "").replace("-int8", "").replace("-int4", "")
        if nombre not in modelos_whisper:
            modelos_whisper.append(nombre)

if not modelos_whisper:
    print("No se encontraron modelos Whisper en la carpeta resources y se descargaran ahora, por favor espere")
    optiModel.optiModeloWhisper()
    print("Descarga Completada, ahora se reiniciara el aplicativo.")
    if sys.platform == "win32":
        os.system("cls")
    else:
        os.system("clear")
    os.execv(sys.executable, [sys.executable] + sys.argv)

print("\nModelos de Whisper disponibles:")
for idx, nombre in enumerate(modelos_whisper, 1):
    if nombre == "large-v3":
        print(f"{idx}: {nombre} no funciona en esta version")
    else:
        print(f"{idx}: {nombre}")
    rongoLineaswhisper+=1

opcion_whisper = input("Selecciona el modelo de Whisper a usar: ").strip()
try:
    modelo_audio = modelos_whisper[int(opcion_whisper) - 1]
except (ValueError, IndexError):
    print("Selección inválida, usando el primero por defecto.")
    rongoLineaswhisper+=1
    modelo_audio = "medium"
for _ in range(rongoLineaswhisper):
    print('\033[F\033[K', end='') 

talkToText.escuchar(device_index, nameDevice, modelo_audio,channels_device)