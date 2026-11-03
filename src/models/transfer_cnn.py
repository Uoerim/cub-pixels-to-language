from torch import nn
from torchvision import models

from src.models.base_encoder import BaseEncoder


class TransferCNN(BaseEncoder):
    def __init__(self, num_classes=200, dropout=0.5, pretrained=True, backbone="resnet18", freeze_backbone=False):
        super().__init__()
        choices = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
            "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
            "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
        }
        if backbone not in choices:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose from {list(choices)}.")
        constructor, weights = choices[backbone]
        network = constructor(weights=weights if pretrained else None)
        self.feature_dim = network.fc.in_features
        self.features = nn.Sequential(*list(network.children())[:-2])
        for layer in self.features.modules():
            if isinstance(layer, nn.ReLU):
                layer.inplace = False
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(1), nn.Dropout(dropout), nn.Linear(self.feature_dim, num_classes))
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.features.requires_grad_(False)
            self.features.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.features.eval()
        return self

    def forward_features(self, images):
        return self.features(images)

    def forward_head(self, features):
        return self.classifier(self.pool(features))

    def get_last_conv_layer(self):
        block = self.features[-1][-1]
        return block.conv3 if hasattr(block, "conv3") else block.conv2
