# training/models/convnext_tiny.py
"""ConvNeXt-Tiny dual-head model."""
import timm
import torch.nn as nn


class AnemiaModel(nn.Module):
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.3, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feature_dim = self.backbone.num_features
        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(256, 1)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f = self.backbone(x)
        return self.regression_head(f), self.classification_head(f)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 3):
        for stage in list(self.backbone.stages)[-n:]:
            for p in stage.parameters():
                p.requires_grad = True
