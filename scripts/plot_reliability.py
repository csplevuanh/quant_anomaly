"""Plot reliability diagram from .npz with 'probs' and 'labels'."""
import argparse, numpy as np
from rnf_experiments.visualization.plots import plot_reliability_diagram

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True)
    p.add_argument('--out', default='figures/reliability.png')
    args = p.parse_args()
    data = np.load(args.npz)
    plot_reliability_diagram(data['probs'], data['labels'], save=args.out)

if __name__ == '__main__':
    main()
