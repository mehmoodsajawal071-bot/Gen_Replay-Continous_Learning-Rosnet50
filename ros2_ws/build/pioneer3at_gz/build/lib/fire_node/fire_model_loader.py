import torch
from torch import nn
import torchvision.models as models

class GenerativeReplayResNet50(nn.Module):
    def __init__(self, initial_classes=6, device='cpu'):
        super().__init__()
        self.device = device
        self.backbone = models.resnet50(pretrained=False)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.feature_dim, initial_classes)
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        features = self.backbone(x)
        return self.classifier(features)

class FireClassifier:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device)

        # Get number of classes from checkpoint
        num_classes = checkpoint['classifier.weight'].shape[0]
        self.model = GenerativeReplayResNet50(initial_classes=num_classes, device=device)

        # Load weights
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
        return probs

