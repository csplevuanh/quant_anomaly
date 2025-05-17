"""Generate ROC curve from a NumPy .npz file with 'scores' and 'labels'."""
import argparse, numpy as np
from rnf_experiments.visualization.plots import plot_roc_curve

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True, help='Path to .npz file containing scores and labels arrays')
    p.add_argument('--out', default='figures/roc_curve.png')
    args = p.parse_args()

    data = np.load(args.npz)
    scores = data['scores']
    labels = data['labels']
    plot_roc_curve(scores, labels, save=args.out)

if __name__ == '__main__':
    main()
