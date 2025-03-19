import os
from torch.utils.data import Dataset
from PIL import Image

class LicensePlateDataset(Dataset):
    def __init__(self, data_dir, data_file_name,transform=None,max_samples=None):
        self.data_dir = data_dir
        self.data_file = os.path.join(data_dir, data_file_name)
        self.transform = transform
        self.images = []
        self.colors = []  

        with open(self.data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break 
                parts = line.strip().split(' ')
                img_path = parts[0]  
                img_path = os.path.join(data_dir, img_path)  
                color = parts[1] 
                self.images.append(img_path)
                self.colors.append(color)
                    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        color = self.colors[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, color