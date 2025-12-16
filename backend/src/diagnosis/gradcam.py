import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _register_hooks(self):
        fwd = self.target_layer.register_forward_hook(self._save_activation)
        bwd = self.target_layer.register_full_backward_hook(self._save_gradient)
        self._hooks = [fwd, bwd]

    def _clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def generate(self, input_tensor, class_idx=None):
        self.gradients = None
        self.activations = None

        self._register_hooks()

        try:
            self.model.zero_grad()
            input_tensor.requires_grad_(True)

            output = self.model(input_tensor)

            if class_idx is None:
                class_idx = output.argmax(dim=1).item()

            one_hot = torch.zeros_like(output)
            one_hot[0, class_idx] = 1

            output.backward(gradient=one_hot, retain_graph=True)

            if self.gradients is None or self.activations is None:
                print("GradCAM failed: missing gradients or activations")
                h, w = input_tensor.shape[2], input_tensor.shape[3]
                return np.zeros((h, w))

            pooled_grads = self.gradients.mean(dim=(0, 2, 3))

            activations = self.activations.detach().clone()
            for i in range(activations.shape[1]):
                activations[:, i, :, :] *= pooled_grads[i]

            heatmap = activations.mean(dim=1).squeeze()
            heatmap = F.relu(heatmap)

            if heatmap.max() != 0:
                heatmap /= heatmap.max()

            return heatmap.cpu().numpy()

        finally:
            self._clear_hooks()


def overlay_heatmap(heatmap, image_path, alpha=0.4, colormap=cv2.COLORMAP_JET):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        img = cv2.resize(img, (224, 224))

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)

        blended = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
        return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f"Overlay error: {e}")
        return None
