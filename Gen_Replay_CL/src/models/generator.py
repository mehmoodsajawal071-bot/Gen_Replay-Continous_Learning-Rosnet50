import torch
import torch.nn as nn

class FeatureReplayGenerator(nn.Module):
    def __init__(self, latent_dim=100, feature_dim=2048, num_tasks=3, num_classes=2, device='cuda'):
        super(FeatureReplayGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        self.device = device
        self.condition_dim = num_tasks + num_classes

        self.generator = nn.Sequential(
            nn.Linear(latent_dim + self.condition_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, feature_dim)
        ).to(self.device)

    def forward(self, noise, condition):
        if noise.device != self.device: noise = noise.to(self.device)
        if condition.device != self.device: condition = condition.to(self.device)
        x = torch.cat([noise, condition], dim=1)
        return self.generator(x)

    def generate_features(self, task_id, class_id, num_samples):
        noise = torch.randn(num_samples, self.latent_dim, device=self.device)
        condition = torch.zeros(num_samples, self.condition_dim, device=self.device)
        condition[:, task_id] = 1
        condition[:, self.num_tasks + class_id] = 1
        with torch.no_grad():
            features = self.forward(noise, condition)
        return features
