# python/edgekit/models/mobilenet.py
#
# MobileNetV2 pre-entraine sur ImageNet, en version "quantifiable" de torchvision (QuantStub,
# DeQuantStub, fuse_model()), pour que le meme modele serve au transfer learning (TP 4), a la
# quantification (TP 5) et au pruning (TP 6).

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights
from torchvision.models.quantization import mobilenet_v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def mobilenet_v2_transfer(n_classes: int, pretrained: bool = True, seed: int | None = 0) -> nn.Module:
    """MobileNetV2 quantifiable dont la derniere couche est remplacee par n_classes sorties."""
    if seed is not None:
        torch.manual_seed(seed)
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights, quantize=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    return model


def set_trainable(model: nn.Module, last_blocks: int = 0) -> int:
    """Gele tout le reseau sauf la tete et les `last_blocks` derniers blocs de `features`.

    Retourne le nombre de parametres entrainables.
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    if last_blocks > 0:
        for block in list(model.features)[-last_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
