from tabpfn import TabPFNClassifier

def tabpfn_gen(pretrained: bool = True):
    """Return TabPFN model wrapper.

    The TabPFN library ships with pre‑trained weights by default."""
    return TabPFNClassifier(device='cuda' if pretrained else 'cpu')
