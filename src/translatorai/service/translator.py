from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForSeq2SeqLM
import datetime
import time
import os

class translatorFull:

    cola_subtitulos = []
    
    def realSub(self, valueText, TXT_FILE,TXT_FILE2):

        
        
        # Ruta local al modelo NLLB-200-distilled-600M INT8
        local_model_path = os.path.abspath("../resources/nllb-200-distilled-600M-int8")
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = OVModelForSeq2SeqLM.from_pretrained(local_model_path)

        inputs = tokenizer(valueText, return_tensors="pt")
        ### translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])
        forced_bos_token_id = tokenizer.convert_tokens_to_ids("<|eng_Latn|>")
        translated_tokens = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)
        # Opcionalmente, puedes cambiar "eng_Latn" por el código de idioma deseado como:
        # es_Latn para español, fr_Latn para francés, etc.
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

        timestamp = time.time()
        self.cola_subtitulos.append((translated_text, timestamp))
        with open(TXT_FILE, "a", encoding="utf-8") as f:
            f.write(f"{translated_text}<br>\n")
        with open(TXT_FILE2, "a", encoding="utf-8") as f2:
                        f2.write(f"{translated_text}\n")
        
    def comprobadorSubtitulos(self,TXT_FILE):
        ahora = time.time()
        self.cola_subtitulos = [(txt, t) for txt, t in self.cola_subtitulos if ahora - t < 2]
        with open(TXT_FILE, "w", encoding="utf-8") as f:
            for txt, _ in self.cola_subtitulos:
                f.write(f"{txt}<br>\n")