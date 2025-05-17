from rnf_experiments.evaluation.metrics import auroc, fpr_at_tpr, ece
import numpy as np

def test_metrics_shapes():
    scores = np.random.rand(100)
    labels = np.random.randint(0, 2, 100)
    assert 0 <= auroc(scores, labels) <= 1
    assert 0 <= fpr_at_tpr(scores, labels) <= 1

def test_ece():
    probs = np.random.rand(200)
    labels = np.random.randint(0, 2, 200)
    val = ece(probs, labels)
    assert 0 <= val <= 1
