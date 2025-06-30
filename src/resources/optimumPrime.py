from optimum.intel.openvino import OVModelForSeq2SeqLM, OVWeightQuantizationConfig
from transformers import AutoTokenizer

model_id = "facebook/nllb-200-distilled-600M"
save_dir4 = "nllb-200-distilled-600M-int4"
save_dir8 = "nllb-200-distilled-600M-int4"

# Configuración de quantización INT4 y INT8
quant_config4 = OVWeightQuantizationConfig(bits=4)
quant_config8 = OVWeightQuantizationConfig(bits=8)

# Descarga y quantiza el modelo
model = OVModelForSeq2SeqLM.from_pretrained(model_id, export=True, quantization_config=quant_config4)
model.save_pretrained(save_dir4)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(save_dir4)

model = OVModelForSeq2SeqLM.from_pretrained(model_id, export=True, quantization_config=quant_config8)
model.save_pretrained(save_dir8)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(save_dir8)

## ct2-transformers-converter --model openai/whisper-large-v2 --output_dir faster-whisper-large-v2-int8 --quantization int8