"""Unified command‑line interface for RN‑F repo."""
import typer, pathlib, importlib, yaml
from rnf_experiments.quantization.ptq import quantize_model
from rnf_experiments.detectors import RNFDetector, LogisticResidual
from rnf_experiments.evaluation.metrics import auroc, fpr_at_tpr, ece
import torch, numpy as np

app = typer.Typer()

@app.command()
def quantize(model: str, bits: int = 4):
    """Quantize a model and save checkpoint."""
    mod = importlib.import_module(f"rnf_experiments.models.{model}")
    mdl_fn = getattr(mod, model.replace('-', '_'))
    mdl = mdl_fn(pretrained=True)
    q_mdl = quantize_model(mdl, bits=bits)
    path = pathlib.Path('models') / f'{model}-int{bits}.pth'
    path.parent.mkdir(exist_ok=True)
    torch.save(q_mdl.state_dict(), path)
    typer.echo(f'Saved quantized model → {path}')

@app.command()
def detect(model: str, detector: str = 'rnf'):
    """Run contamination detection (dummy example)."""
    feats = torch.randn(512, 64)
    labels = torch.randint(0, 2, (512,))
    det_cls = RNFDetector if detector == 'rnf' else LogisticResidual
    det = det_cls(feats.size(1))
    scores = det(feats).detach().numpy()
    path = pathlib.Path('logs') / f'{model}_{detector}.npz'
    path.parent.mkdir(exist_ok=True)
    np.savez(path, scores=scores, labels=labels.numpy())
    typer.echo(f'Saved scores to {path}')

@app.command()
def evaluate(npz: str):
    data = np.load(npz)
    scores, labels = data['scores'], data['labels']
    print('AUROC:', auroc(scores, labels))
    print('FPR@95:', fpr_at_tpr(scores, labels))
    probs = 1 / (1 + np.exp(-scores))
    print('ECE:', ece(probs, labels))

@app.command()
def run_cfg(config: str):
    """Run experiment from YAML config."""
    with open(config) as f:
        cfg = yaml.safe_load(f)
    typer.echo(cfg)

if __name__ == "__main__":
    app()
