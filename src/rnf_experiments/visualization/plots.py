"""Visualization helpers used for Figures 4–6."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_roc_curve(scores, labels, title="ROC Curve", save: str | None = None):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'AUROC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    _finalise(save)

def plot_residual_histogram(residuals, labels, bins=50, title="Residual Histogram", save: str | None = None):
    plt.figure()
    for lab, col in [(0, 'Predicted Clean'), (1, 'Predicted Contaminated')]:
        plt.hist(residuals[labels == lab], bins=bins, alpha=0.5, label=col, density=True)
    plt.xlabel('Residual Norm')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    _finalise(save)

def plot_reliability_diagram(probabilities, labels, n_bins=15, title="Reliability Diagram", save: str | None = None):
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(probabilities, bins) - 1
    bin_acc = [labels[binids == i].mean() if np.any(binids == i) else 0 for i in range(n_bins)]
    bin_conf = [probabilities[binids == i].mean() if np.any(binids == i) else (bins[i]+bins[i+1])/2 for i in range(n_bins)]

    plt.figure()
    plt.plot([0, 1], [0, 1], '--')
    plt.bar(bins[:-1] + (bins[1]-bins[0])/2, bin_acc, width=(bins[1]-bins[0])*0.9, align='center', alpha=0.7)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(title)
    _finalise(save)

def _finalise(save: str | None):
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, dpi=300, bbox_inches='tight')
        print(f'[RNF‑Viz] 📈 Saved figure to {save}')
    else:
        plt.show()
