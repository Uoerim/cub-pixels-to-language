import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.visualization.plots import denormalize


class GradCAM:
    def __init__(self, model, target_layer=None, output_size=(224, 224)):
        self.model = model
        self.output_size = output_size
        self.activations = None
        self.gradients = None
        layer = target_layer if target_layer is not None else model.get_last_conv_layer()
        self.handles = [
            layer.register_forward_hook(self._capture_activations),
            layer.register_full_backward_hook(self._capture_gradients),
        ]

    def _capture_activations(self, module, inputs, output):
        self.activations = output.detach()

    def _capture_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, images, target_classes=None):
        was_training = self.model.training
        self.model.eval()
        self.activations = None
        self.gradients = None
        try:
            with torch.enable_grad():
                device = next(self.model.parameters()).device
                inputs = images.detach().clone().to(device).requires_grad_(True)
                self.model.zero_grad(set_to_none=True)
                logits = self.model(inputs)
                if target_classes is None:
                    targets = logits.argmax(dim=1)
                else:
                    targets = torch.as_tensor(target_classes, device=device, dtype=torch.long)
                    if targets.ndim == 0:
                        targets = targets.repeat(logits.size(0))
                    if targets.shape != (logits.size(0),):
                        raise ValueError("Provide one target class per image.")
                    if (targets < 0).any() or (targets >= logits.size(1)).any():
                        raise ValueError("Target class is outside the model's class range.")
                logits.gather(1, targets[:, None]).sum().backward()
                if self.activations is None or self.gradients is None:
                    raise ValueError("The target layer must participate in the model's forward pass.")
                weights = self.gradients.mean(dim=(2, 3), keepdim=True)
                maps = (weights * self.activations).sum(dim=1, keepdim=True).relu()
                maps = F.interpolate(maps, size=self.output_size, mode="bilinear", align_corners=False)
                minimum = maps.amin(dim=(2, 3), keepdim=True)
                maximum = maps.amax(dim=(2, 3), keepdim=True)
                maps = (maps - minimum) / (maximum - minimum).clamp_min(1e-8)
                return maps[:, 0].detach().cpu(), logits.detach().cpu()
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def overlay_heatmap(image, heatmap, alpha=0.4):
    original = denormalize(image)
    values = np.asarray(heatmap, dtype=np.float32)
    if values.shape != original.shape[:2]:
        values = F.interpolate(torch.from_numpy(values)[None, None], size=original.shape[:2], mode="bilinear", align_corners=False)[0, 0].numpy()
    color = matplotlib.colormaps["jet"](np.clip(values, 0, 1))[..., :3]
    return np.clip((1 - alpha) * original + alpha * color, 0, 1)


def save_gradcam_examples(model, loader, output_dir="outputs/gradcam", num_samples=8, class_names=None, max_scan_samples=None):
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = list(class_names or getattr(loader.dataset, "class_names", []))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    candidates = {"correct": [], "wrong": []}
    quotas = {"correct": (num_samples + 1) // 2, "wrong": num_samples // 2}
    scanned = 0
    try:
        with torch.no_grad():
            for images, labels in loader:
                predictions = model(images.to(device)).argmax(dim=1).cpu()
                for image, label, prediction in zip(images, labels, predictions):
                    if max_scan_samples is not None and scanned >= max_scan_samples:
                        break
                    category = "correct" if int(label) == int(prediction) else "wrong"
                    if len(candidates[category]) < num_samples:
                        candidates[category].append((scanned, image.detach().cpu().clone(), int(label)))
                    scanned += 1
                    if all(len(candidates[key]) >= quotas[key] for key in quotas):
                        break
                if all(len(candidates[key]) >= quotas[key] for key in quotas) or (max_scan_samples is not None and scanned >= max_scan_samples):
                    break
        selected = [(category, item) for category in quotas for item in candidates[category][:quotas[category]]]
        extras = [(category, item) for category in quotas for item in candidates[category][quotas[category]:]]
        selected.extend(extras[:max(0, num_samples - len(selected))])
        manifest = []
        with GradCAM(model) as gradcam:
            for category, (index, image, label) in selected:
                maps, logits = gradcam(image.unsqueeze(0))
                probabilities = logits.softmax(dim=1)[0]
                prediction = int(probabilities.argmax())
                confidence = float(probabilities[prediction])
                actual_name = names[label] if label < len(names) else str(label)
                predicted_name = names[prediction] if prediction < len(names) else str(prediction)
                figure, axes = plt.subplots(1, 2, figsize=(9, 4))
                axes[0].imshow(denormalize(image))
                axes[0].set_title(f"True: {actual_name}", fontsize=10)
                axes[1].imshow(overlay_heatmap(image, maps[0].numpy()))
                axes[1].set_title(f"Predicted: {predicted_name}\nConfidence: {confidence:.1%}", fontsize=10)
                for axis in axes:
                    axis.axis("off")
                figure.tight_layout()
                path = output_dir / f"{category}_{index:05d}_true_{label}_pred_{prediction}.png"
                figure.savefig(path, dpi=160)
                plt.close(figure)
                manifest.append({"file": path.name, "category": category, "label": label, "prediction": prediction, "confidence": confidence})
        summary = {"scanned": scanned, "requested": num_samples, "saved": len(manifest), "examples": manifest}
        (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return [output_dir / entry["file"] for entry in manifest]
    finally:
        model.train(was_training)
