# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
from typing import Tuple

from torch import nn

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE
from flash.image.backbones import catch_url_error

if _TORCHVISION_AVAILABLE:
    import torchvision

RESNET_MODELS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d"]

IMAGE_CLASSIFIER_BACKBONES = FlashRegistry("backbones")

if _TORCHVISION_AVAILABLE:

    def _fn_resnet(model_name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
        model: nn.Module = getattr(torchvision.models, model_name, None)(pretrained)
        backbone = nn.Sequential(*list(model.children())[:-2])
        num_features = model.fc.in_features
        return backbone, num_features

    for model_name in RESNET_MODELS:
        IMAGE_CLASSIFIER_BACKBONES(
            fn=catch_url_error(partial(_fn_resnet, model_name)),
            name=model_name,
            namespace="vision",
            package="torchvision",
            type="resnet"
        )
