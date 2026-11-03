from abc import ABC, abstractmethod

from torch import nn


class BaseEncoder(nn.Module, ABC):
    @abstractmethod
    def forward_features(self, images):
        raise NotImplementedError

    @abstractmethod
    def forward_head(self, features):
        raise NotImplementedError

    def forward(self, images):
        return self.forward_head(self.forward_features(images))

    @abstractmethod
    def get_last_conv_layer(self):
        raise NotImplementedError
