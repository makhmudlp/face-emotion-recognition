import torch
import torch.nn as nn
from torchvision import models

class EmotionResNet(nn.Module):
    def __init__(self, num_classes=7, dropout=0.4):
        super(EmotionResNet, self).__init__()

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        with torch.no_grad():
            self.resnet.conv1.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self.freeze_backbone()

    def freeze_backbone(self):
        for name, param in self.resnet.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        print("Backbone frozen — only training head")

    def unfreeze_last_blocks(self):
        for name, param in self.resnet.named_parameters():
            if any(x in name for x in ["layer3", "layer4", "fc"]):
                param.requires_grad = True
        print("Unfroze layer3, layer4 + head")

    def unfreeze_all(self):
        for param in self.resnet.parameters():
            param.requires_grad = True
        print("Unfroze entire network")

    def count_trainable(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable:,} / {total:,} parameters")

    def forward(self, x):
        return self.resnet(x)