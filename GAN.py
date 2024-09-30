import os
import glob
from PIL import Image
import pandas as pd
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch

class_map = pd.read_csv('/data/ephemeral/home/Taegyun/map_clsloc.txt', sep= ' ')
train = pd.read_csv('/data/ephemeral/home/data/train.csv')
class_map_True = class_map[class_map['class_num'].isin(train['class_name'])]
class_nums = class_map_True['class_num'].values
class_names = class_map_True['class_name'].values
class_dict = dict(zip(class_nums, class_names))
train_dir = '/data/ephemeral/home/data/train/'
save_dir = '/data/ephemeral/home/Taegyun/GaN/'

# 모델을 전역 변수로 선언
pipe_img2img = None

# 모델을 한 번만 로드하고 전역으로 사용
def init_model():
    global pipe_img2img
    if pipe_img2img is None:
        pipe_img2img = AutoPipelineForImage2Image.from_pretrained(
            "dreamlike-art/dreamlike-photoreal-2.0", torch_dtype=torch.float16, use_safetensors=True
        ).to("cuda")


# 이미지 처리 함수 (모델은 전역 변수로 접근)
def process_image(num, name, img_path):
    try:
        init_image = load_image(img_path)
        prompt = f"Based on existing images, create a sketch of a {name} in black and white, with a size of (224, 224)."
        generator = torch.Generator(device="cuda").manual_seed(328)
        # 이미지 생성
        generated_image = pipe_img2img(prompt, image=init_image, generator=generator).images[0]

        # 저장할 파일 경로 설정
        image_filename = os.path.join(save_dir, f"GaN_{num}_{os.path.basename(img_path)}")  # 파일 이름 설정
        generated_image.save(image_filename)  # 이미지 저장
        print(f"Saved: {image_filename}")
    except Exception as e:
        print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    os.makedirs(save_dir, exist_ok=True)

    # 모델을 부모 프로세스에서 한 번만 로드
    init_model()

    cnt = 0
    for num, name in class_dict.items():
        cnt += 1
        class_path = glob.glob(os.path.join(train_dir + num + '/*'))

        # 순차적으로 이미지 처리
        for img_path in class_path:
            process_image(num, name, img_path)

        if cnt % 10 == 0:
            print(f'{num} 끝났어요')

    print("All images have been processed and saved.")
