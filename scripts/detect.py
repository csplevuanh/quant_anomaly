import argparse, torch
from rnf_experiments.detectors.rnf import RNFDetector
from rnf_experiments.evaluation.metrics import auroc
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--detector', default='rnf')
    args = parser.parse_args()

    # Dummy demo
    feats = torch.randn(512, 64)
    labels = torch.randint(0, 2, (512,))
    detector = RNFDetector(feats.size(1))
    scores = detector(feats).detach().numpy()
    print(f"AUROC ≈ {auroc(scores, labels.numpy()):.3f}")

if __name__ == "__main__":
    main()
