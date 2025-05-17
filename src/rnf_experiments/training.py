"""Training utilities for RN‑F detector."""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

def train_detector(detector, dataset, epochs=5, batch_size=128, lr=1e-3, device='cuda'):
    detector.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    optim = torch.optim.Adam(detector.parameters(), lr=lr)
    crit = torch.nn.BCEWithLogitsLoss()

    detector.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for feats, labels in tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}'):
            feats, labels = feats.to(device), labels.to(device).float()
            logits = detector(feats)
            loss = crit(logits, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * feats.size(0)
        print(f'Epoch {epoch+1} loss: {epoch_loss/len(dataset):.4f}')
    return detector
