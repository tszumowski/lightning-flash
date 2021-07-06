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
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from flash.core.data.data_module import DataModule
from flash.core.data.data_source import DefaultDataKeys, DefaultDataSources, PathsDataSource
from flash.core.data.process import Preprocess
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE
from flash.image.classification.transforms import default_transforms

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS


class ImageClassificationPathsDataSource(PathsDataSource):
    """The ``ImageClassificationPathsDataSource`` is a :class:`~flash.core.data.data_source.PathsDataSource` which loads
    PIL images from files using ``torchvision``.
    """

    def __init__(self, labels: Optional[Sequence[str]] = None):
        super().__init__(extensions=IMG_EXTENSIONS, labels=labels)

    def load_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        sample[DefaultDataKeys.INPUT] =\
            default_loader(sample[DefaultDataKeys.INPUT])
        return sample


class ImageClassificationPreprocess(Preprocess):
    """The ``ImageClassificationPreprocess`` is a :class:`~flash.core.data.process.Preprocess` for loading and
    transforming data for image classification tasks.
    """

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        labels: Optional[Sequence[str]] = None,
        image_size: Tuple[int, int] = (64, 64),
    ):
        self.image_size = image_size

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={DefaultDataSources.FOLDERS: ImageClassificationPathsDataSource(labels=labels)},
            default_data_source=DefaultDataSources.FOLDERS
        )

    def default_transforms(self) -> Optional[Dict[str, Callable]]:
        return default_transforms(self.image_size)

    def get_state_dict(self) -> Dict[str, Any]:
        return self.transforms

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        return cls(**state_dict)


class ImageClassificationData(DataModule):
    """The ``ImageClassificationData`` class is a :class:`~flash.core.data.data_module.DataModule`` for use in image
    classification tasks.
    """

    preprocess_cls = ImageClassificationPreprocess
