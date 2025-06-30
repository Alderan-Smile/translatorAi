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

print("\nSelecciona el modelo de traducción a utilizar:")
print("1: NLLB-200-distilled-600M INT8 (Optimum/OpenVINO, solo Intel CPU o GPU)")
print("2: NLLB-200-distilled-600M INT4 (Optimum/OpenVINO, solo Intel CPU o GPU)")
print("3: NLLB-200-distilled-600M ORIGINAL (PyTorch, soporta GPU Nvidia)")

opcion = input("Ingresa el número de modelo (1/2/3): ").strip()
if opcion == "1":
    modelo_traductor = "INT8"
elif opcion == "2":
    modelo_traductor = "INT4"
elif opcion == "3":
    modelo_traductor = "ORIGINAL"
else:
    print("Opción no válida, usando modelo ORIGINAL por defecto.")
    modelo_traductor = "ORIGINAL"
for _ in range(5):
    print('\033[F\033[K', end='') 
print(f"Modelo seleccionado: {modelo_traductor}\n")

talkToText.escuchar(device_index, nameDevice, modelo_traductor)