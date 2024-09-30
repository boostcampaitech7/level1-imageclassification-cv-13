import torch

TRAIN_DATA_DIR = "./data/train"
TRAIN_DATA_INFO_FILE = "./data/train.csv"
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10
MODEL_NAME = "resnet18"
NUM_CLASSES = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# MoDEL_NAME

# hugging face
# clip_mlp : openai/clip-vit-large-patch14

# timm 
# resnet18 : resnet18   
# Vit : vit_base_patch16_224
# swin_T : swin_base_patch4_window7_224