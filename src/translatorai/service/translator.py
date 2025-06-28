from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MarianMTModel, MarianTokenizer
import os

class translatorFull:
    
    def realSub(self,valueText):
        src_text = [
            f">>eng<< {valueText}"
        ]
    
        local_model_path = "./src/resources/opus-multi-MMT"

        tokenizer = MarianTokenizer.from_pretrained(local_model_path)
        model = MarianMTModel.from_pretrained(local_model_path)
        translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))



        for t in translated:
            print( tokenizer.decode(t, skip_special_tokens=True) )
        