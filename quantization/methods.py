import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from awq import AutoAWQForCausalLM
from awq.quantize.quantizer import AWQConfig
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

class QuantizationMethod:
    """Base class for quantization methods."""

    def __init__(self, config: Dict):
        self.config = config
        self.name = config["name"]
        self.bits = config["bits"]
        self.library = config["library"]

    def quantize(
        self,
        model_name: str,
        save_dir: Optional[str] = None,
        **kwargs
    ) -> Tuple[Union[AutoModelForCausalLM, ORTModelForCausalLM], AutoTokenizer]:
        """Quantize a model and return the quantized model + tokenizer."""
        raise NotImplementedError

    def get_model_size_mb(self, model) -> float:
        """Get model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / (1024 * 1024)

class FP16Quantizer(QuantizationMethod):
    """FP16 baseline (no quantization)."""

    def quantize(self, model_name: str, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=self.config.get("device_map", "auto"),
            trust_remote_code=True
        )
        return model, tokenizer

class BitsAndBytesQuantizer(QuantizationMethod):
    """Quantization using bitsandbytes (INT8/INT4)."""

    def quantize(self, model_name: str, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.bits == 8:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map=self.config.get("device_map", "auto"),
                trust_remote_code=True
            )
        elif self.bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=self.config["config"].get("bnb_4bit_use_double_quant", True),
                bnb_4bit_quant_type=self.config["config"].get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=self.config.get("device_map", "auto"),
                trust_remote_code=True
            )
        else:
            raise ValueError(f"Unsupported bits: {self.bits}")

        return model, tokenizer

class GPTQQuantizer(QuantizationMethod):
    """Quantization using GPTQ."""

    def quantize(self, model_name: str, save_dir: Optional[str] = None, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create save directory if not provided
        if save_dir is None:
            save_dir = f"quantized_models/{model_name.replace('/', '_')}_gptq_{self.bits}bit"

        quantize_config = BaseQuantizeConfig(
            bits=self.config["config"]["bits"],
            group_size=self.config["config"].get("group_size", 128),
            desc_act=self.config["config"].get("desc_act", False)
        )

        model = AutoGPTQForCausalLM.from_pretrained(
            model_name,
            quantize_config=quantize_config,
            trust_remote_code=True
        )

        # Save quantized model
        model.save_quantized(save_dir)
        tokenizer.save_pretrained(save_dir)

        # Load quantized model
        model = AutoGPTQForCausalLM.from_quantized(
            save_dir,
            device_map=self.config.get("device_map", "auto"),
            trust_remote_code=True
        )

        return model, tokenizer

class AWQQuantizer(QuantizationMethod):
    """Quantization using AWQ."""

    def quantize(self, model_name: str, save_dir: Optional[str] = None, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if save_dir is None:
            save_dir = f"quantized_models/{model_name.replace('/', '_')}_awq_{self.bits}bit"

        # Quantize
        quant_config = AWQConfig(
            bits=self.config["config"]["bits"],
            group_size=self.config["config"].get("group_size", 128)
        )

        model = AutoAWQForCausalLM.from_pretrained(
            model_name,
            quant_config=quant_config,
            trust_remote_code=True
        )

        # Save quantized model
        model.save_quantized(save_dir)
        tokenizer.save_pretrained(save_dir)

        # Load quantized model
        model = AutoAWQForCausalLM.from_quantized(
            save_dir,
            device_map=self.config.get("device_map", "auto"),
            trust_remote_code=True
        )

        return model, tokenizer

class OptimumQuantizer(QuantizationMethod):
    """Quantization using Hugging Face Optimum (ONNX)."""

    def quantize(self, model_name: str, save_dir: Optional[str] = None, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if save_dir is None:
            save_dir = f"quantized_models/{model_name.replace('/', '_')}_optimum_{self.bits}bit"

        # Export to ONNX
        model = ORTModelForCausalLM.from_pretrained(
            model_name,
            export=True,
            trust_remote_code=True
        )

        # Quantize
        if self.bits == 8:
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, perchannel=False)
        else:
            raise ValueError(f"Optimum only supports 8-bit quantization (got {self.bits})")

        model.quantize(quantization_config=qconfig)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        # Reload
        model = ORTModelForCausalLM.from_pretrained(
            save_dir,
            trust_remote_code=True
        )

        return model, tokenizer

def get_quantizer(config: Dict) -> QuantizationMethod:
    """Factory method to get the appropriate quantizer."""
    method = config["method"]
    if method == "fp16":
        return FP16Quantizer(config)
    elif method == "int8" or method == "int4":
        return BitsAndBytesQuantizer(config)
    elif method == "gptq":
        return GPTQQuantizer(config)
    elif method == "awq":
        return AWQQuantizer(config)
    elif method == "optimum-int8":
        return OptimumQuantizer(config)
    else:
        raise ValueError(f"Unsupported quantization method: {method}")
