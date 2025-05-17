"""Post‑Training Quantization utilities."""
import torch
import bitsandbytes as bnb

def quantize_model(model: torch.nn.Module, bits: int = 4) -> torch.nn.Module:
    """Naïve GPTQ‑style quantization using bitsandbytes Int<n> parameters."""
    if bits not in (2, 4, 8):
        raise ValueError("Only 2‑, 4‑ or 8‑bit quantization supported")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        int_param = bnb.nn.Int8Params(param.data, quant_type=f'int{bits}')
        setattr(model, name, int_param)
    return model
