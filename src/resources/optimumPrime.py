import os
import subprocess
import argparse

# This project uses PyTorch/OpenVINO paths; disabling TF avoids Keras 3 compatibility import errors.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

class optimizadorModelos:

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

    def _resolver_ruta(self, ruta_relativa):
        return os.path.abspath(os.path.join(self.base_dir, ruta_relativa))

    def optiModeloTraductor(self):
        # Lazy imports avoid loading heavy backends when this module is imported.
        from optimum.intel.openvino import OVModelForSeq2SeqLM, OVWeightQuantizationConfig
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_id = "facebook/nllb-200-distilled-600M"
        save_dirO = self._resolver_ruta("../resources/nllb-200-distilled-600M")
        save_dir4 = self._resolver_ruta("../resources/nllb-200-distilled-600M-int4")
        save_dir8 = self._resolver_ruta("../resources/nllb-200-distilled-600M-int8")
        model_hf = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        model_hf.save_pretrained(save_dirO)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(save_dirO)
        quant_config4 = OVWeightQuantizationConfig(bits=4)
        model_int4 = OVModelForSeq2SeqLM.from_pretrained(model_id, export=True, quantization_config=quant_config4)
        model_int4.save_pretrained(save_dir4)
        quant_config8 = OVWeightQuantizationConfig(bits=8)
        model_int8 = OVModelForSeq2SeqLM.from_pretrained(model_id, export=True, quantization_config=quant_config8)
        model_int8.save_pretrained(save_dir8)

        model_id2 = "nvidia/parakeet-rnnt-1.1b"
        save_dir1 = self._resolver_ruta("../resources/parakeet-rnnt-1.1b")
        save_dir3 = self._resolver_ruta("../resources/parakeet-rnnt-1.1b-int8")
        save_dir4 = self._resolver_ruta("../resources/parakeet-rnnt-1.1b-float16")
        model_hf = AutoModelForSeq2SeqLM.from_pretrained(model_id2)
        model_hf.save_pretrained(save_dir1)
        tokenizer = AutoTokenizer.from_pretrained(model_id2)
        tokenizer.save_pretrained(save_dir1)
        quant_config2 = OVWeightQuantizationConfig(bits=8)
        model_int8 = OVModelForSeq2SeqLM.from_pretrained(model_id2, export=True, quantization_config=quant_config2)
        model_int8.save_pretrained(save_dir3)
        quant_config3 = OVWeightQuantizationConfig(bits=16)
        model_f16 = OVModelForSeq2SeqLM.from_pretrained(model_id2, export=True, quantization_config=quant_config3)
        model_f16.save_pretrained(save_dir4)

    def optiModeloWhisper(self):
        comandos =[
            ["ct2-transformers-converter",
            "--model","openai/whisper-large-v3-turbo",
            "--output_dir",self._resolver_ruta("../resources/faster-whisper-large-v3-turbo-int8"),
            "--quantization","int8"]
            ,["ct2-transformers-converter",
            "--model","openai/whisper-large-v3",
            "--output_dir",self._resolver_ruta("../resources/faster-whisper-large-v3-int8"),
            "--quantization","int8"]
            ,["ct2-transformers-converter",
            "--model","openai/whisper-large-v2",
            "--output_dir",self._resolver_ruta("../resources/faster-whisper-large-v2-int8"),
            "--quantization","int8"]
            ,["ct2-transformers-converter",
            "--model","openai/whisper-large",
            "--output_dir",self._resolver_ruta("../resources/faster-whisper-large-int8"),
            "--quantization","int8"]
            ,["ct2-transformers-converter",
            "--model","openai/whisper-medium",
            "--output_dir",self._resolver_ruta("../resources/faster-whisper-medium-int8"),
            "--quantization","int8"]
            ,["ct2-transformers-converter",
            "--model","openai/whisper-small",
            "--output_dir",self._resolver_ruta("../resources/faster-whisper-small-int8"),
            "--quantization","int8"]
            ,["ct2-transformers-converter",
            "--model","openai/whisper-base",
            "--output_dir",self._resolver_ruta("../resources/faster-whisper-base-int8"),
            "--quantization","int8"]
            ,["ct2-transformers-converter",
            "--model","distil-whisper/distil-large-v3",
            "--output_dir",self._resolver_ruta("../resources/faster-distil-whisper-large-v3-float16"),
            "--quantization","float16"]
            ,["ct2-transformers-converter",
            "--model","distil-whisper/distil-large-v2",
            "--output_dir",self._resolver_ruta("../resources/faster-distil-whisper-large-v2-float16"),
            "--quantization","float16"]
            ##,["ct2-transformers-converter",
            ##"--model","openai/whisper-turbo",
            ##"--output_dir","../resources/faster-whisper-turbo-int8",
            ##"--quantization","int8"]
        ]

        for cmd in comandos:
            subprocess.run(cmd,check=True)

    def optiModeloWhisperFaltantes(self):
        modelos = [
            ("openai/whisper-large-v3-turbo", self._resolver_ruta("../resources/faster-whisper-large-v3-turbo-int8")),
            ("openai/whisper-large-v3", self._resolver_ruta("../resources/faster-whisper-large-v3-int8")),
            ("openai/whisper-large-v2", self._resolver_ruta("../resources/faster-whisper-large-v2-int8")),
            ("openai/whisper-large", self._resolver_ruta("../resources/faster-whisper-large-int8")),
            ("openai/whisper-medium", self._resolver_ruta("../resources/faster-whisper-medium-int8")),
            ("openai/whisper-small", self._resolver_ruta("../resources/faster-whisper-small-int8")),
            ("openai/whisper-base", self._resolver_ruta("../resources/faster-whisper-base-int8")),
            ("distil-whisper/distil-large-v3", self._resolver_ruta("../resources/faster-distil-whisper-large-v3-float16")),
            ("distil-whisper/distil-large-v2", self._resolver_ruta("../resources/faster-distil-whisper-large-v2-float16")),
        ]

        faltantes = [(model_id, output_dir) for model_id, output_dir in modelos if not os.path.isdir(output_dir)]
        if not faltantes:
            print("No hay modelos Whisper faltantes.")
            return

        for model_id, output_dir in faltantes:
            print(f"Descargando modelo faltante: {model_id}")
            cmd = [
                "ct2-transformers-converter",
                "--model", model_id,
                "--output_dir", output_dir,
                "--quantization", "int8",
            ]
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga y cuantiza modelos para TranslatorAI")
    parser.add_argument(
        "accion",
        nargs="?",
        default="whisper-faltantes",
        choices=["whisper", "whisper-faltantes", "traductor"],
        help="whisper: descarga todos, whisper-faltantes: solo faltantes, traductor: descarga y cuantiza NLLB",
    )
    args = parser.parse_args()

    opt = optimizadorModelos()
    if args.accion == "whisper":
        opt.optiModeloWhisper()
    elif args.accion == "whisper-faltantes":
        opt.optiModeloWhisperFaltantes()
    else:
        opt.optiModeloTraductor()