import torch
from sklearn.metrics import roc_auc_score

def auroc(scores, labels):
    return roc_auc_score(labels, scores)

def fpr_at_tpr(scores, labels, target_tpr=0.95):
    import numpy as np
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(labels, scores)
    try:
        return np.interp(target_tpr, tpr, fpr)
    except Exception:
        return 1.0

def ece(probs, labels, n_bins=15):
    import numpy as np
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(probs, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if not mask.any():
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += abs(acc - conf) * mask.mean()
    return ece
