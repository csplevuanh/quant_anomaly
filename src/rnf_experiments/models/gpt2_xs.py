from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "stas/tiny-gpt2"  # community replica of GPT2‑XS

def tinystories_gpt2_xs(pretrained: bool = True):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME if pretrained else None)
    return mdl, tok
