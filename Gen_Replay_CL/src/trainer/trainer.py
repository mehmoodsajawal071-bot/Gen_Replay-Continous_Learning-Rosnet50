import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class GenerativeReplayTrainer:
    def __init__(self, model, generator, device, tracker, datasets_config, lr=1e-4, replay_lambda=0.5):
        self.model = model
        self.generator = generator
        self.device = device
        self.tracker = tracker
        self.datasets_config = datasets_config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.replay_lambda = replay_lambda
        self.past_task_data = {}

    def _evaluate_task(self, loader, task_name):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model.get_task_output(images, task_name)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return 0.0, (correct / total if total > 0 else 0.0)

    def train_task(self, train_loader, val_loader, task_name, epochs):
        self.model.register_task(task_name)
        if task_name not in self.model.task_output_slices:
            num_classes = len(self.datasets_config[task_name]['classes'])
            if not self.model.task_output_slices:
                self.model.task_output_slices[task_name] = slice(0, num_classes)
            else:
                self.model.expand_classifier(num_classes, task_name)

        for epoch in range(epochs):
            self.model.train()
            for images, labels in tqdm(train_loader, desc=f"Task {task_name} Epoch {epoch+1}"):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model.get_task_output(images, task_name)
                loss_current = self.criterion(outputs, labels)

                loss_replay = torch.tensor(0.0, device=self.device)
                if self.past_task_data:
                    for past_name in self.past_task_data:
                        past_config = self.datasets_config[past_name]
                        gen_feats = self.generator.generate_features(past_config['task_id'], 0, images.size(0))
                        # ... Replay Logic from notebook ...

                (loss_current + self.replay_lambda * loss_replay).backward()
                self.optimizer.step()

            _, val_acc = self._evaluate_task(val_loader, task_name)
            print(f"Epoch {epoch+1} Val Acc: {val_acc:.4f}")

        self.past_task_data[task_name] = True
