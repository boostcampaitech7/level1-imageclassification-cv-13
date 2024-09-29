import torch
from torch.utils.data import DataLoader
from data_loader import CustomDataset, load_data, split_data  
from models.models import CustomModel
from models.model_CLIP import CLIP_MLP
from model_selector import ModelSelector
from trainer import Trainer
from utils import accuracy
from transform import Transform
import config
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn

total_params = 397
params_per_group = 20
def set_param_groups(model, group_index, params_per_group):
    cnt = 0
    for param in model.parameters():
        # 해당 그룹의 파라미터만 고정 해제
        if group_index * params_per_group <= cnt < (group_index + 1) * params_per_group:
            param.requires_grad = True  # 해제
        else:
            param.requires_grad = False  # 고정
        cnt += 1
# 총 그룹 개수 계산 (나머지 파라미터 포함)
num_groups = (total_params + params_per_group - 1) // params_per_group  # 그룹 개수 계산 (올림)

def main():
    # 데이터 로드 및 분할
    train_info = load_data(config.TRAIN_DATA_INFO_FILE)
    train_df, val_df = split_data(train_info)
    
    # 데이터셋 및 데이터 로더 설정
    train_transform = Transform(is_train=True)
    val_transform = Transform(is_train=False)

    train_dataset = CustomDataset(config.TRAIN_DATA_DIR, train_df, train_transform, is_inference=False)
    val_dataset = CustomDataset(config.TRAIN_DATA_DIR, val_df, val_transform, is_inference=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 모델 로드 및 프리징
    model_selector = ModelSelector(model_name=config.MODEL_NAME, num_classes=config.NUM_CLASSES)
    model = model_selector.get_model()
    model = model.to(config.DEVICE)
    model.load_state_dict(torch.load('/data/ephemeral/home/Dongook/model_dw/Clip_rop.pt'))
    for group_index in range(num_groups):
        # 옵티마이저와 학습 설정
        model.load_state_dict(torch.load('/data/ephemeral/home/jb_train_result/Clip_lr.pt'))
        set_param_groups(model,group_index,params_per_group)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-6, weight_decay=0.01)
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        # Trainer 설정 및 학습
        trainer = Trainer(model, optimizer, criterion, config.DEVICE, early_stop=False, patience_limit=5)
        
        trainer.train(train_loader, val_loader, epochs=5)
        # 최종 모델 저장
        model.load_state_dict(torch.load('/data/ephemeral/home/jb_train_result/Clip_lr.pt'))
        torch.save(trainer.model.state_dict(), f'/data/ephemeral/home/jb_train_result/Clip_fine{group_index}.pt')


if __name__ == "__main__":
    main()
