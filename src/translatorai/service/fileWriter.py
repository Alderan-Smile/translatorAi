import os

class controlArchivos:

    def compruebaCarpetas(self,Carpeta):
        os.makedirs(os.path.dirname(Carpeta), exist_ok=True)

    def compruebaArchivos(self,TXT_FILE):
        with open(TXT_FILE, "a", encoding="utf-8"):
            pass

    def escribeArchivos(self,TXT_FILE,TEXTP):
        with open(TXT_FILE, "a", encoding="utf-8") as f:
            f.write(TEXTP)

    def borraArchivos(self,TXT_FILE,cola_subtitulos):
        with open(TXT_FILE, "w", encoding="utf-8") as f:
            for txt, _ in cola_subtitulos:
                f.write(f"{txt}<br>\n")