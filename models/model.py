import torch.nn as nn
from torchvision import models


def get_model(num_classes):

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace classifier
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model