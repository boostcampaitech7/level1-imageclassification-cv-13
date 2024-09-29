
import torch
from torch.utils.data import DataLoader, ConcatDataset
from data_loader import CustomDataset, load_data, split_data  
from models.models import CustomModel
from models.model_CLIP import CLIP_MLP
from model_selector import ModelSelector
from trainer import Trainer
from utils import accuracy
from transform import Transform
import config
from torch.optim.lr_scheduler import  ReduceLROnPlateau


def main():
    train_info = load_data(config.TRAIN_DATA_INFO_FILE)
    train_df, val_df = split_data(train_info)
    
    train_transform = Transform(is_train = True)
    val_transform = Transform(is_train = False)

    aug_dataset = CustomDataset(config.TRAIN_DATA_DIR, train_df, train_transform, is_inference=False)
    train_dataset = CustomDataset(config.TRAIN_DATA_DIR, train_df, val_transform, is_inference = False)
    val_dataset = CustomDataset(config.TRAIN_DATA_DIR, val_df, val_transform, is_inference=False)


    train_dataset = ConcatDataset([train_dataset, aug_dataset])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers = 4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)


    
    model_selector = ModelSelector(model_name=config.MODEL_NAME, num_classes=config.NUM_CLASSES)
    model = model_selector.get_model()
    model = model.to(config.DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    trainer = Trainer(model, optimizer, criterion, config.DEVICE, scheduler, early_stop=True, patience_limit=5)
    trainer.train(train_loader, val_loader, config.EPOCHS)

if __name__ == "__main__":
    main()
