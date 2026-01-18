import torch
import os
from models.resnet_gr import GenerativeReplayResNet50
from models.generator import FeatureReplayGenerator
from data.loader import create_dataloaders
from trainer.trainer import GenerativeReplayTrainer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_PATH = "classfication-datasets/"

datasets_config = {
    'Violence': {'path': os.path.join(BASE_PATH, 'Violence'), 'classes': ['non_violence', 'violence'], 'task_id': 0},
    'myhelmet': {'path': os.path.join(BASE_PATH, 'myhelmet'), 'classes': ['No-Helmet', 'Helmet'], 'task_id': 1},
    'FireDetection': {'path': os.path.join(BASE_PATH, 'FireDetection'), 'classes': ['fire', 'non-fire'], 'task_id': 2},
}

def main():
    dataloaders = create_dataloaders(datasets_config)
    model = GenerativeReplayResNet50(initial_classes=2, device=DEVICE)
    generator = FeatureReplayGenerator(num_tasks=len(datasets_config), device=DEVICE)

    # Passing None for tracker if you haven't defined it yet
    trainer = GenerativeReplayTrainer(model, generator, DEVICE, None, datasets_config)

    for task_name in datasets_config.keys():
        print(f"\n--- Starting Task: {task_name} ---")
        loader_info = dataloaders[task_name]
        trainer.train_task(loader_info['train'], loader_info['val'], task_name, epochs=10)

if __name__ == "__main__":
    main()
