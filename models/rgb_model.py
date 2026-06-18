import torch.nn as nn
from torchvision import models

def create_rgb_model():

    model = models.efficientnet_b0(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(
            model.classifier[1].in_features,
            10
        )
    )

    return model