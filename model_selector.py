from models.models import CustomModel
from models.model_CLIP import CLIP_MLP




class ModelSelector:
    def __init__(self, model_name: str, num_classes: int):
        """
        모델 선택기를 초기화합니다.
        :param model_name: 사용할 모델 이름
        :param num_classes: 모델의 출력 클래스 수
        """
        self.model_name = model_name
        self.num_classes = num_classes

    def get_model(self):
        """
        모델 이름에 따라 적절한 모델을 반환합니다.
        :return: 선택된 모델 인스턴스
        """
        if self.model_name == "resnet18":
            return CustomModel("resnet18", self.num_classes)
        
        elif self.model_name == "resnet50":
            return CustomModel("resnet50", self.num_classes)
        
        elif self.model_name == "Vit":
            return CustomModel("vit_base_patch16_224", self.num_classes)
        
        elif self.model_name == "swin_T":
            return CustomModel("swin_base_patch4_window7_224", self.num_classes)
        
        elif self.model_name == "clip_mlp":
            return CLIP_MLP("openai/clip-vit-large-patch14", self.num_classes)
        
        else:
            raise ValueError(f"지원하지 않는 모델 이름입니다: {self.model_name}")
