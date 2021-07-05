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
from typing import Callable, Dict, Tuple

import torch
from torch import nn
from torch.utils.data._utils.collate import default_collate

from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.transforms import ApplyToKeys
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    import torchvision


def default_transforms(image_size: Tuple[int, int]) -> Dict[str, Callable]:
    """The default transforms for image classification: resize the image, convert the image and targets to a tensor,
    collate the batch.
    """
    return {
        "pre_tensor_transform": ApplyToKeys(DefaultDataKeys.INPUT, torchvision.transforms.Resize(image_size)),
        "to_tensor_transform": nn.Sequential(
            ApplyToKeys(DefaultDataKeys.INPUT, torchvision.transforms.ToTensor()),
            ApplyToKeys(DefaultDataKeys.TARGET, torch.as_tensor),
        ),
        "collate": default_collate,
    }
