import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights



class InceptionCustom(nn.Module):
    def __init__(self, num_classes: int):
        super(InceptionCustom, self).__init__()
        self.base = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits = True)
        self.base.fc = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(2048, num_classes)

    def freeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = False
        
    def unfreeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.base(x)
        if isinstance(x, tuple):
            x = x[0]
        x = self.classifier(x)
        return x
