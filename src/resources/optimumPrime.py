from optimum.intel.openvino import OVModelForSeq2SeqLM, OVWeightQuantizationConfig
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
save_dirO = "nllb-200-distilled-600M"
save_dir4 = "nllb-200-distilled-600M-int4"
save_dir8 = "nllb-200-distilled-600M-int4"

# 1. Descargar y guardar el modelo original HuggingFace
model_hf = AutoModelForSeq2SeqLM.from_pretrained(model_id)
model_hf.save_pretrained(save_dirO)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(save_dirO)

# 2. Quantización INT4
quant_config4 = OVWeightQuantizationConfig(bits=4)
model_int4 = OVModelForSeq2SeqLM.from_pretrained(model_id, export=True, quantization_config=quant_config4)
model_int4.save_pretrained(save_dir4)

# 3. Quantización INT8
quant_config8 = OVWeightQuantizationConfig(bits=8)
model_int8 = OVModelForSeq2SeqLM.from_pretrained(model_id, export=True, quantization_config=quant_config8)
model_int8.save_pretrained(save_dir8)

## ct2-transformers-converter --model openai/whisper-large-v2 --output_dir faster-whisper-large-v2-int8 --quantization int8
## ct2-transformers-converter --model openai/whisper-medium --output_dir faster-whisper-medium-int8 --quantization int8