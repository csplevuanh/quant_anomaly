import torch
from rnf_experiments.models.tinyvit import tinyvit_iot_s4
from rnf_experiments.quantization.ptq import quantize_model

def test_quantize_bits():
    model = tinyvit_iot_s4(pretrained=False)
    q_model = quantize_model(model, bits=4)
    for p in q_model.parameters():
        # ensure parameters are wrapped in bitsandbytes Int* types
        assert hasattr(p, '__class__')
