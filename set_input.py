import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, filenames, image_size=128, K=4):
        self.filenames = filenames
        self.image_size = image_size
        self.K = K
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAdjustSharpness(sharpness_factor=0.1),
            transforms.ColorJitter(brightness=0.05, contrast=0.05),  # ColorJitter는 흑백에 영향을 미치지 않음
            transforms.Resize((image_size + 16, image_size + 16)),  # Crop + wiggle offset
            transforms.RandomCrop((image_size, image_size)),
            transforms.ToTensor(),  # HWC to CHW로 변환 및 정규화
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Read the image
        image = Image.open(self.filenames[idx]).convert("L")  # 'L'로 흑백 모드로 변환

        # Apply transformations
        image = self.transform(image)

        # Create downsampled feature
        downsampled = F.interpolate(image.unsqueeze(0), size=(self.image_size // self.K, self.image_size // self.K), mode='area')

        feature = downsampled.squeeze(0)
        label = image

        return feature, label

# Usage
def setup_inputs(filenames, batch_size=32, image_size=128, K=4):
    dataset = ImageDataset(filenames, image_size=image_size, K=K)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader

# Example usage
filenames = [os.path.join('path_to_images', f) for f in os.listdir('path_to_images') if f.endswith('.jpg')]
dataloader = setup_inputs(filenames)

for features, labels in dataloader:
    # Use features and labels in your training loop
    pass
