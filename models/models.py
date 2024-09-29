
import torch
import torch.nn as nn
import timm

class CustomModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super(CustomModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)