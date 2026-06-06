import torch
import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)


class EfficientNet512(nn.Module):
    """
        EfficientNet-B0 backbone with a 512-dimensional embedding layer
        and binary classification head for deepfake detection.
    """
    def __init__(self):
        super().__init__()

        self.backbone = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        in_features = (
            self.backbone.classifier[1].in_features
        )

        self.backbone.classifier = nn.Identity()

        self.embedding = nn.Sequential(
            nn.Linear(
                in_features,
                512
            ),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(
            512,
            1
        )

    def forward(
        self,
        x,
        return_embedding=False
    ):
        features = self.backbone(x)

        embedding = self.embedding(
            features
        )

        if return_embedding:
            return embedding

        return self.classifier(
            embedding
        )