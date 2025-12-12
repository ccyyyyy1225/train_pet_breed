from PIL import Image
import torch
from torchvision import transforms

def get_eval_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def preprocess_image(pil_img: Image.Image, device="cpu", img_size=224):
    tfm = get_eval_transform(img_size)
    x = tfm(pil_img.convert("RGB")).unsqueeze(0)
    return x.to(device)
