import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from trainer import Trainer  

import timm

# CLIP 모델 정의
class Clip(nn.Module):
    def __init__(self, image_encoder):
        super(Clip, self).__init__()  
        self.clip = image_encoder
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

# 모델들을 로드하는 함수
def load_models():
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    image_encoder = clip.vision_model
    clip_model = Clip(image_encoder)
    clip_model.load_state_dict(torch.load('/data/ephemeral/home/Dongook/model_dw/Clip_rop.pt'))
    # clip_model = torch.load('/data/ephemeral/home/Dongook/model_dw/Clip_rop.pt')
    
    cnn_model = timm.create_model('tf_efficientnet_b3', pretrained=False, num_classes=500)
    cnn_model.load_state_dict(torch.load('/data/ephemeral/home/Dongook/model_dw/effib3_83.79.pt'))
    # cnn_model = torch.load('/data/ephemeral/home/Dongook/model_dw/effib3_83.79.pt')

    swin_model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=500)
    swin_model.load_state_dict(torch.load('/data/ephemeral/home/Dongook/model_dw/swin_feed.pt'))
    # swin_model = torch.load('/data/ephemeral/home/Dongook/model_dw/swin_feed.pt')

    res_model = timm.create_model('resnet50d', pretrained=False, num_classes=500)
    res_model.load_state_dict(torch.load('/data/ephemeral/home/Dongook/model_dw/res50d_e120.pt'))
    # res_model = torch.load('/data/ephemeral/home/Dongook/model_dw/res50d_e120.pt')

    return clip_model, cnn_model, swin_model, res_model

# 앙상블 모델 정의
class AdaptiveEnsemble(nn.Module):
    def __init__(self, clip_model, cnn_model, swin_model, res_model):
        super().__init__()
        self.clip = clip_model
        self.cnn = cnn_model
        self.swin = swin_model
        self.res = res_model
        
        # 각 모델의 가중치 초기값
        self.weights = nn.Parameter(torch.tensor([0.3, 0.2, 0.3, 0.2]))  

    def forward(self, x):
        clip_y = self.clip(x)
        cnn_y = self.cnn(x)
        swin_y = self.swin(x)
        res_y = self.res(x)
        
        # 가중치를 softmax로 정규화
        normalized_weights = F.softmax(self.weights, dim=0)
        
        # 가중치에 따라 앙상블
        ensemble_y = (normalized_weights[0] * clip_y + 
                      normalized_weights[1] * cnn_y + 
                      normalized_weights[2] * swin_y + 
                      normalized_weights[3] * res_y)
        
        return ensemble_y

# 앙상블 모델 학습 함수
def train_ensemble_model(ensemble_model, train_loader, val_loader, device, epochs=5):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW([ensemble_model.weights], lr=5e-4, weight_decay=1e-4)

    # ReduceLROnPlateau 스케줄러 설정
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Trainer 클래스 인스턴스 생성
    trainer = Trainer(model=ensemble_model, 
                      optimizer=optimizer, 
                      criterion=criterion, 
                      device=device,
                      scheduler = scheduler, 
                      early_stop=True, 
                      patience_limit=3)

    # 학습 시작
    trainer.train(train_loader, val_loader, epochs=epochs)
