import torch
import torch.nn as nn
import torchvision.models as models

def build_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def load_torchscript_model(model_path: str, device: str = "cpu"):
    device = torch.device(device)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model
