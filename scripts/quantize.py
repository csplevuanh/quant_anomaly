import argparse, importlib, torch

from rnf_experiments.quantization.ptq import quantize_model
from rnf_experiments.models import tinyvit, gpt2_xs, tabpfn

MODELS = {
    "tinyvit-iot-s4": tinyvit.tinyvit_iot_s4,
    "gpt2-xs": gpt2_xs.tinystories_gpt2_xs,
    "tabpfn": tabpfn.tabpfn_gen,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS.keys(), required=True)
    parser.add_argument('--bits', type=int, default=4)
    args = parser.parse_args()

    model_fn = MODELS[args.model]
    model = model_fn()[0] if isinstance(model_fn(), tuple) else model_fn()
    model = quantize_model(model, bits=args.bits)
    torch.save(model.state_dict(), f"models/{args.model}-int{args.bits}.pth")
    print(f"[RNF] ✅ Saved quantized model to models/{args.model}-int{args.bits}.pth")

if __name__ == "__main__":
    main()
