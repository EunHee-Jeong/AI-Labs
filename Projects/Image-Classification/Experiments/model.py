import torch.nn as nn
import torchvision.models as models
from torchvision.models import get_model_weights, get_model

class BaseModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        # 모델 로드 (자동으로 가중치도 가져옴)
        try:
            weights = get_model_weights(model_name).DEFAULT
        except ValueError:
            raise ValueError(f"❌ '{model_name}' is not a valid torchvision model name.")

        self.backbone = get_model(model_name, weights=weights)

        self.feature_dim = self._get_feature_dim(model_name)
        self._remove_classifier(model_name)

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head = nn.Linear(self.feature_dim, num_classes)

    def _get_feature_dim(self, model_name):
        if hasattr(self.backbone, 'fc'):
            return self.backbone.fc.in_features
        elif hasattr(self.backbone, 'classifier'):
            last_layer = self.backbone.classifier[-1]
            return last_layer.in_features if hasattr(last_layer, 'in_features') else last_layer.in_channels
        else:
            raise NotImplementedError("❌ Unknown model structure.")

    def _remove_classifier(self, model_name):
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()
        else:
            raise NotImplementedError("❌ Unknown model structure.")

    def forward(self, x):
        x = self.backbone(x)
        if x.ndim == 4:
            x = self.pooling(x)
            x = self.flatten(x)
        x = self.head(x)
        return x
