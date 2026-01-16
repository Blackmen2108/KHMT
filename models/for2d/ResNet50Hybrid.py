import torch
import torch.nn as nn
import torchvision.models as models


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
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


class ResNet50Hybrid(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, freeze_backbone=False, use_attention=True):
        super().__init__()

        # Load ResNet50 backbone
        base_model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Extract layers
        self.stem = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        )
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        
        # Optional freezing
        if freeze_backbone:
            for param in base_model.parameters():
                param.requires_grad = False

        # Attention layer (SE block)
        self.use_attention = use_attention
        if use_attention:
            self.attention = SEBlock(2048, reduction=16)
        else:
            self.attention = nn.Identity()

        # Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Feature extraction
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Attention enhancement
        x = self.attention(x)

        # Global pooling and head
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x