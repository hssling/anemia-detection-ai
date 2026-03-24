# training/models/efficientnet_b4.py
"""EfficientNet-B4 dual-head model for hemoglobin regression + anemia classification."""
import timm
import torch
import torch.nn as nn


class AnemiaModel(nn.Module):
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.3, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feature_dim = self.backbone.num_features
        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(256, 1)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        return self.regression_head(features), self.classification_head(features)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 3):
        for block in list(self.backbone.blocks)[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in self.backbone.conv_head.parameters():
            param.requires_grad = True
        for param in self.backbone.bn2.parameters():
            param.requires_grad = True
