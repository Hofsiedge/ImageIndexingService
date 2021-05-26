import os, torch
from pathlib import Path
from PIL import Image


class PascalVOC2012(torch.utils.data.Dataset):
    def __init__(self, location: str, transform, device=None):
        self.image_dir = Path(location)
        self.names     = os.listdir(self.image_dir)
        self.transform = transform
        self.device    = device or 'cpu'
    
    def __len__(self) -> int:
        return len(self.names)
    
    def __getitem__(self, idx) -> torch.Tensor:
        image = Image.open(self.image_dir / self.names[idx])
        return self.transform(image)
