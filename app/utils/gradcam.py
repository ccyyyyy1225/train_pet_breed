import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """Minimal Grad-CAM for ResNet-like models.
    Works with TorchScript model ONLY if it preserves module names; therefore, this file is optional.
    For maximum compatibility in Streamlit, you may disable Grad-CAM if it fails.
    """
    def __init__(self, model, target_layer_name="layer4"):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _find_layer(self):
        # TorchScript may not expose named_modules the same way; this may fail.
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                return module
        return None

    def _register_hooks(self):
        layer = self._find_layer()
        if layer is None:
            return

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    def generate(self, x, class_idx: int):
        if self._find_layer() is None:
            return None

        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward()

        if self.gradients is None or self.activations is None:
            return None

        # weights: GAP over gradients
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_cam_on_image(pil_img, cam, alpha=0.4):
    if cam is None:
        return None
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (1 - alpha) * img + alpha * heatmap
    overlay = overlay.astype(np.uint8)
    return overlay
