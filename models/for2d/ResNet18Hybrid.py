import torch
import torch.nn as nn
import torchvision.models as models

# ---------- Squeeze-and-Excitation (Attention) ----------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ---------- Vision Transformer Encoder ----------
class TransformerEncoder2D(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, depth=2):
        super(TransformerEncoder2D, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(depth)
        ])

    def forward(self, x):
        # flatten spatial dims into sequence (B, C, H, W) -> (B, HW, C)
        b, c, h, w = x.size()
        seq = x.view(b, c, h * w).permute(0, 2, 1)
        for layer in self.layers:
            seq = layer(seq)
        # reshape back
        seq = seq.permute(0, 2, 1).view(b, c, h, w)
        return seq

# ---------- Full Model ----------
class ResNet18Hybrid(nn.Module):
    def __init__(self, num_classes=1, weights='IMAGENET1K_V1', freeze_backbone=True):
        super(ResNet18Hybrid, self).__init__()

        # Load pretrained ResNet18 backbone
        self.backbone = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])  # remove avgpool + fc
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # Add attention + transformer blocks
        self.se_block = SEBlock(channels=512, reduction=16)
        self.transformer = TransformerEncoder2D(embed_dim=512, num_heads=8, depth=2)

        # Classification head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.se_block(x)
        x = self.transformer(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x