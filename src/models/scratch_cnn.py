from torch import nn

from src.models.base_encoder import BaseEncoder


class ScratchCNN(BaseEncoder):
    def __init__(self, num_classes=200, dropout=0.5):
        super().__init__()
        channels = [3, 64, 128, 256, 512, 512]
        stages = []
        for in_channels, out_channels in zip(channels, channels[1:]):
            stages.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ))
        self.features = nn.Sequential(*stages)
        self.feature_dim = channels[-1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward_features(self, images):
        return self.features(images)

    def forward_head(self, features):
        return self.classifier(self.pool(features))

    def get_last_conv_layer(self):
        return self.features[-1][0]
