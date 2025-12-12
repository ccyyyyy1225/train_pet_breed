import torch
import torch.nn as nn
import torchvision.models as models

def build_mobilenetv2(num_classes: int, pretrained: bool = False) -> nn.Module:
    model = models.mobilenet_v2(weights=None if not pretrained else models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def load_state_dict_model(model_path: str, class_count: int, device: str = "cpu"):
    device = torch.device(device)
    model = build_mobilenetv2(num_classes=class_count, pretrained=False)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model

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
