"""PyTorch Dataset wrapper for the tri‑modal M5Product dataset."""
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import json
import pandas as pd

class M5ProductDataset(Dataset):
    def __init__(self, root: Path, split='train', transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.img_dir = self.root / 'images' / split
        self.text_path = self.root / f'{split}_text.jsonl'
        self.tab_path = self.root / f'{split}_tabular.parquet'

        self.meta = pd.read_parquet(self.tab_path)
        with open(self.text_path) as f:
            self.text_data = [json.loads(line) for line in f]

        assert len(self.meta) == len(self.text_data)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        img_path = self.img_dir / f'{idx}.jpg'
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        text = self.text_data[idx]['description']
        tabular = self.meta.iloc[idx].to_numpy(dtype='float32')
        return image, text, tabular
