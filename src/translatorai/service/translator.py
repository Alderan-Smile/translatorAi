from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datetime
import time

class translatorFull:
    
    def realSub(self, valueText):

        cola_subtitulos = []
        
        # Ruta local al modelo NLLB-200-distilled-600M INT8
        local_model_path = "./src/resources/nllb-200-distilled-600M-int8"
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)

        inputs = tokenizer(valueText, return_tensors="pt")
        translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"]) 
        # Opcionalmente, puedes cambiar "eng_Latn" por el código de idioma deseado como:
        # es_Latn para español, fr_Latn para francés, etc.
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)


        fecha_hora = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        TXT_FILE = "./src/translatorai/subtitle/subt.txt"
        TXT_FILE2 = f"./src/translatorai/subtitle/backuptrad_{fecha_hora}.txt"
        with open(TXT_FILE, "a", encoding="utf-8") as f:
            f.write(f"{translated_text}<br>\n")
        with open(TXT_FILE2, "a", encoding="utf-8") as f2:
                        f2.write(f"{translated_text}\n")
        print(translated_text)
        
        ahora = time.time()

        cola_subtitulos = [(txt, t) for txt, t in cola_subtitulos if ahora - t < 2]
        with open(TXT_FILE, "w", encoding="utf-8") as f:
            for txt, _ in cola_subtitulos:
                f.write(f"{txt}<br>\n")