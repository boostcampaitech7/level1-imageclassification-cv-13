import torch
from torch.utils.data import DataLoader
from ensemble import load_models, AdaptiveEnsemble, train_ensemble_model
from data_loader import CustomDataset, load_data, split_data   
from transform import Transform  
import config

def main():

    train_info = load_data(config.TRAIN_DATA_INFO_FILE)
    train_df, val_df = split_data(train_info)
    
    train_transform = Transform(is_train = True)
    val_transform = Transform(is_train = False)

    train_dataset = CustomDataset(config.TRAIN_DATA_DIR, train_df, train_transform, is_inference=False)
    val_dataset = CustomDataset(config.TRAIN_DATA_DIR, val_df, val_transform, is_inference=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers = 4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)


    # 모델 로드
    clip_model, cnn_model, swin_model, res_model = load_models()

    # 앙상블 모델 생성
    ensemble_model = AdaptiveEnsemble(clip_model, cnn_model, swin_model, res_model).to(config.DEVICE)

    # 학습 시작
    epochs = config.EPOCHS
    train_ensemble_model(ensemble_model, train_loader, val_loader, config.DEVICE, epochs)

if __name__ == "__main__":
    main()
