import torch
import timm
import torch.nn as nn
from transformers import CLIPModel



class CLIP_MLP(nn.Module):
    def __init__(self, num_classes: int):
        super(CLIP_MLP, self).__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").vision_model

        
        self.mlp = nn.Sequential(
            nn.Linear(1024, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 500)
        )

    def forward(self, images):
        with torch.no_grad():
            image_features = self.clip(images).last_hidden_state
        
        y = self.mlp(image_features[:, 0, :])
        return y