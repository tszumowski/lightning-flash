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
from typing import Any, Dict

from flash.core.data.data_module import DataModule
from flash.core.data.process import Preprocess


class ImageClassificationPreprocess(Preprocess):
    """The ``ImageClassificationPreprocess`` is a :class:`~flash.core.data.process.Preprocess` for loading and
    transforming data for image classification tasks.
    """

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
