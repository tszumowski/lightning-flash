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
import functools
import os
import urllib.error
import warnings
from functools import partial
from typing import Tuple

from pytorch_lightning.utilities import rank_zero_warn
from torch import nn

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _BOLTS_AVAILABLE, _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

if _BOLTS_AVAILABLE:
    if os.getenv("WARN_MISSING_PACKAGE") == "0":
        with warnings.catch_warnings(record=True) as w:
            pass
    else:
        pass

ROOT_S3_BUCKET = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com"

RESNET_MODELS = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d"]

OBJ_DETECTION_BACKBONES = FlashRegistry("backbones")


def catch_url_error(fn):

    @functools.wraps(fn)
    def wrapper(*args, pretrained=False, **kwargs):
        try:
            return fn(*args, pretrained=pretrained, **kwargs)
        except urllib.error.URLError:
            result = fn(*args, pretrained=False, **kwargs)
            rank_zero_warn(
                "Failed to download pretrained weights for the selected backbone. The backbone has been created with"
                " `pretrained=False` instead. If you are loading from a local checkpoint, this warning can be safely"
                " ignored.", UserWarning
            )
            return result

    return wrapper


if _TORCHVISION_AVAILABLE:

    def _fn_resnet_fpn(
        model_name: str,
        pretrained: bool = True,
        trainable_layers: bool = True,
        **kwargs,
    ) -> Tuple[nn.Module, int]:
        backbone = resnet_fpn_backbone(model_name, pretrained=pretrained, trainable_layers=trainable_layers, **kwargs)
        return backbone, 256

    for model_name in RESNET_MODELS:
        OBJ_DETECTION_BACKBONES(
            fn=catch_url_error(partial(_fn_resnet_fpn, model_name)),
            name=model_name,
            package="torchvision",
            type="resnet-fpn"
        )
