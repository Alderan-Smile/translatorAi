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
            ("openai/whisper-large-v3-turbo", self._resolver_ruta("../resources/faster-whisper-large-v3-turbo-int8"), "int8"),
            ("openai/whisper-large-v3", self._resolver_ruta("../resources/faster-whisper-large-v3-int8"), "int8"),
            ("openai/whisper-large-v2", self._resolver_ruta("../resources/faster-whisper-large-v2-int8"), "int8"),
            ("openai/whisper-large", self._resolver_ruta("../resources/faster-whisper-large-int8"), "int8"),
            ("openai/whisper-medium", self._resolver_ruta("../resources/faster-whisper-medium-int8"), "int8"),
            ("openai/whisper-small", self._resolver_ruta("../resources/faster-whisper-small-int8"), "int8"),
            ("openai/whisper-base", self._resolver_ruta("../resources/faster-whisper-base-int8"), "int8"),
            ("distil-whisper/distil-large-v3", self._resolver_ruta("../resources/faster-distil-whisper-large-v3-float16"), "float16"),
            ("distil-whisper/distil-large-v2", self._resolver_ruta("../resources/faster-distil-whisper-large-v2-float16"), "float16"),
        ]

        faltantes = [(mid, out, quant) for mid, out, quant in modelos if not os.path.isdir(out)]
        if not faltantes:
            print("No hay modelos Whisper faltantes.")
            return

        for model_id, output_dir, quantization in faltantes:
            print(f"Descargando modelo faltante: {model_id} ({quantization})")
            cmd = [
                "ct2-transformers-converter",
                "--model", model_id,
                "--output_dir", output_dir,
                "--quantization", quantization,
            ]
            subprocess.run(cmd, check=True)

    def optiModeloTraductorFaltantes(self):
        from optimum.intel.openvino import OVModelForSeq2SeqLM, OVWeightQuantizationConfig
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        modelos = [
            (
                "facebook/nllb-200-distilled-600M",
                self._resolver_ruta("../resources/nllb-200-distilled-600M"),
                None,
            ),
            (
                "facebook/nllb-200-distilled-600M",
                self._resolver_ruta("../resources/nllb-200-distilled-600M-int4"),
                OVWeightQuantizationConfig(bits=4),
            ),
            (
                "facebook/nllb-200-distilled-600M",
                self._resolver_ruta("../resources/nllb-200-distilled-600M-int8"),
                OVWeightQuantizationConfig(bits=8),
            ),
        ]

        for model_id, output_dir, quant_config in modelos:
            if os.path.isdir(output_dir):
                print(f"Ya existe, omitiendo: {output_dir}")
                continue

            print(f"Descargando/cuantizando: {model_id} -> {os.path.basename(output_dir)}")
            if quant_config is None:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
                model.save_pretrained(output_dir)
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                tokenizer.save_pretrained(output_dir)
            elif quant_config == "float16":
                # OpenVINO no soporta OVWeightQuantizationConfig(bits=16);
                # exportar sin cuantización de pesos deja el modelo en float32/fp16 nativo.
                model = OVModelForSeq2SeqLM.from_pretrained(
                    model_id, export=True
                )
                model.save_pretrained(output_dir)
            else:
                model = OVModelForSeq2SeqLM.from_pretrained(
                    model_id, export=True, quantization_config=quant_config
                )
                model.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descarga y cuantiza modelos para TranslatorAI")
    parser.add_argument(
        "accion",
        nargs="?",
        default="whisper-faltantes",
        choices=["whisper", "whisper-faltantes", "traductor", "traductor-faltantes"],
        help=(
            "whisper: descarga todos los modelos Whisper, "
            "whisper-faltantes: solo los Whisper faltantes, "
            "traductor: descarga y cuantiza NLLB (todos), "
            "traductor-faltantes: solo versiones NLLB faltantes"
        ),
    )
    args = parser.parse_args()

    opt = optimizadorModelos()
    if args.accion == "whisper":
        opt.optiModeloWhisper()
    elif args.accion == "whisper-faltantes":
        opt.optiModeloWhisperFaltantes()
    elif args.accion == "traductor-faltantes":
        opt.optiModeloTraductorFaltantes()
    else:
        opt.optiModeloTraductor()