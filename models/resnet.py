
import torch
import torch.nn as nn
from torchvision import models

class EmotionResNet(nn.Module):
    def __init__(self, num_classes=7, dropout=0.4):
        super(EmotionResNet, self).__init__()

        # Load pretrained ResNet18
        # pretrained weights from ImageNet
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Fix 1: Change first conv layer to accept 1 channel instead of 3 ---
        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1,                          # 1 input channel (grayscale)
            original_conv.out_channels, # keep 64 output channels
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        # Smart weight initialization:
        # Average the 3 RGB channel weights into 1 grayscale channel
        # Better than random initialization — keeps pretrained knowledge
        with torch.no_grad():
            self.resnet.conv1.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )

        # Fix 2: Replace final classifier head ---
        # Original ResNet head: Linear(512 → 1000) for ImageNet
        # Our head: Linear(512 → 7) for emotions
        in_features = self.resnet.fc.in_features  # 512 for ResNet18
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # Phase 1: Freeze everything except the head ---
        self.freeze_backbone()

    def freeze_backbone(self):
        """Freeze all layers except the final classifier head"""
        for name, param in self.resnet.named_parameters():
            if "fc" not in name:        # fc = our custom head
                param.requires_grad = False
        print("Backbone frozen — only training head")

    def unfreeze_last_blocks(self):
        """Phase 2: Unfreeze layer3, layer4 + head"""
        for name, param in self.resnet.named_parameters():
            if any(x in name for x in ["layer3", "layer4", "fc"]):
                param.requires_grad = True
        print("Unfroze layer3, layer4 + head")

    def unfreeze_all(self):
        """Phase 3: Unfreeze everything"""
        for param in self.resnet.parameters():
            param.requires_grad = True
        print("Unfroze entire network")

    def count_trainable(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable:,} / {total:,} parameters")

    def forward(self, x):
        return self.resnet(x)

