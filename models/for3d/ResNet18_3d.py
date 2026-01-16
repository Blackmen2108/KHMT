# models/for3d/swin3d.py
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

class ResNet18_3D(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False, num_classes=1, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = r3d_18(pretrained=(pretrained and hasattr(r3d_18, "__call__")))
        feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(feat, num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        feats = self.backbone(x)
        if feats.dim() > 2:
            feats = feats.view(feats.size(0), -1)
        return self.head(feats)
