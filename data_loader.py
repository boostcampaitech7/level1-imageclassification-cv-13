
import os
import cv2
import torch
import pandas as pd
import numpy as np
from typing import Tuple, Callable, Union
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def load_data(csv_file):
    # 학습 데이터의 정보가 들어있는 CSV 파일을 읽기
    train_info = pd.read_csv(csv_file)
    return train_info

def split_data(train_info, test_size=0.2):
    train_df, val_df = train_test_split(
        train_info, 
        test_size=test_size, 
        stratify=train_info['target']
    )
    return train_df, val_df



class CustomDataset(Dataset):
    def __init__(self, root_dir: str, info_df: pd.DataFrame, transform: Callable, is_inference: bool = False):
        self.root_dir = root_dir
        self.transform = transform 
        self.is_inference = is_inference
        self.image_paths = info_df['image_path'].tolist()

        if not self.is_inference:
            self.targets = info_df['target'].tolist()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        img_path = os.path.join(self.root_dir, self.image_paths[index])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform.fill_white(image)
        image = self.transform(image = image)

        if self.is_inference:
            return image
        else:
            target = self.targets[index]
            return image, target
