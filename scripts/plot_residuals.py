"""Plot residual histogram from .npz with 'residuals' and 'labels'."""
import argparse, numpy as np
from rnf_experiments.visualization.plots import plot_residual_histogram

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True)
    p.add_argument('--out', default='figures/residual_hist.png')
    args = p.parse_args()
    data = np.load(args.npz)
    plot_residual_histogram(data['residuals'], data['labels'], save=args.out)

if __name__ == '__main__':
    main()
