import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_classes=1, weights='IMAGENET1K_V1'):
        super(ResNet50, self).__init__()
        # Load pretrained ResNet18
        self.resnet50 = models.resnet50(weights=weights)
        
        # Replace the fully connected layer with a custom classification layer
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_features, 1)
        )

    def forward(self, x):
        return self.resnet50(x)

# To test the model definition:
if __name__ == "__main__":
    image = torch.randn(4, 3, 64, 64)

    model = ResNet50()

    # input image to model
    output = model(image)