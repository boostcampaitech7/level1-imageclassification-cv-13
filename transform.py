import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import cv2

class Transform:
    def __init__(self, is_train: bool = True):
        common_transforms = [
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            ToTensorV2()
        ]
        
        if is_train:
            self.transform = A.Compose(
                [
                    A.Affine(scale=(0.5, 1.5), p=0.5),
                    A.CoarseDropout(max_holes=4, max_height=30, max_width=30, fill_value=255, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.LongestMaxSize(256),
                    A.PadIfNeeded(256, 256, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)),
                    A.RandomCrop(224, 224,p=1.0),
                ] + common_transforms
            )
        else:
            self.transform = A.Compose(common_transforms)


    def fill_white(self, img):
        edges_r = cv2.Canny(img[:, :, 0], 50, 150)
        edges_g = cv2.Canny(img[:, :, 1], 50, 150)
        edges_b = cv2.Canny(img[:, :, 2], 50, 150)

        edges_combined = np.maximum(np.maximum(edges_r, edges_g), edges_b)

        kernel = np.ones((3, 3), np.uint8)  
        closed_edges = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel)

        filled_image_rgb = cv2.cvtColor(closed_edges, cv2.COLOR_GRAY2RGB)

        inverted_edges = cv2.bitwise_not(closed_edges)

        filled_image_rgb = np.full_like(img, 255)  
        filled_image_rgb[inverted_edges == 0] = [0, 0, 0]
        
        return filled_image_rgb

    def __call__(self, image) -> torch.Tensor:
        if not isinstance(image, np.ndarray):
            raise TypeError("Image should be a NumPy array (OpenCV format).")
        
        transformed = self.transform(image=image)  # 이미지에 설정된 변환을 적용
        return transformed['image']