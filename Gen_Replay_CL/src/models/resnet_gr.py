import torch
import torch.nn as nn
import torchvision.models as models

class GenerativeReplayResNet50(nn.Module):
    def __init__(self, initial_classes=2, device='cuda'):
        super(GenerativeReplayResNet50, self).__init__()
        self.device = device
        self.backbone = models.resnet50(pretrained=True)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.feature_dim, initial_classes).to(self.device)
        self.registered_tasks = []
        self.task_output_slices = {}
        self.to(self.device)

    def expand_classifier(self, new_classes, task_name):
        old_weight = self.classifier.weight.data.clone()
        old_bias = self.classifier.bias.data.clone()
        old_out_features = self.classifier.out_features
        new_out_features = old_out_features + new_classes

        new_classifier = nn.Linear(self.feature_dim, new_out_features).to(self.device)
        new_classifier.weight.data[:old_out_features] = old_weight
        new_classifier.bias.data[:old_out_features] = old_bias

        nn.init.xavier_uniform_(new_classifier.weight.data[old_out_features:])
        nn.init.zeros_(new_classifier.bias.data[old_out_features:])

        self.classifier = new_classifier
        self.task_output_slices[task_name] = slice(old_out_features, new_out_features)

    def get_task_output(self, x, task_name):
        if x.device != self.device: x = x.to(self.device)
        features = self.backbone(x)
        full_output = self.classifier(features)
        if task_name in self.task_output_slices:
            return full_output[:, self.task_output_slices[task_name]]
        return full_output

    def register_task(self, task_name):
        if task_name not in self.registered_tasks:
            self.registered_tasks.append(task_name)

    def forward(self, x):
        if x.device != self.device: x = x.to(self.device)
        features = self.backbone(x)
        return self.classifier(features)
