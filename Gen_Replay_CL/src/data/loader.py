import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ResearchDataset(Dataset):
    def __init__(self, task_config, split='train', transform=None):
        self.task_config = task_config
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = task_config['classes']
        self.task_id = task_config['task_id']
        self._load_data()

    def _load_data(self):
        split_path = os.path.join(self.task_config['path'], self.split)
        if not os.path.exists(split_path): return
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(split_path, class_name)
            if os.path.exists(class_path):
                image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for img_file in image_files:
                    self.images.append(os.path.join(class_path, img_file))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def create_dataloaders(datasets_config, batch_size=16):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataloaders = {}
    for task_name, config in datasets_config.items():
        train_ds = ResearchDataset(config, 'train', train_transform)
        val_ds = ResearchDataset(config, 'valid', val_transform)
        dataloaders[task_name] = {
            'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
            'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2),
            'config': config
        }
    return dataloaders
