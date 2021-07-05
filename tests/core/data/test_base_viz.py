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
from typing import Any, List, Sequence

import numpy as np
import pytest
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.core.data.base_viz import BaseVisualization
from flash.core.utilities.imports import _PIL_AVAILABLE
from tests.helpers.utils import _IMAGE_TESTING

if _PIL_AVAILABLE:
    from PIL import Image


def _rand_image():
    return Image.fromarray(np.random.randint(0, 255, (196, 196, 3), dtype="uint8"))


class CustomBaseVisualization(BaseVisualization):

    def __init__(self):
        super().__init__()

        self.show_load_sample_called = False
        self.show_pre_tensor_transform_called = False
        self.show_to_tensor_transform_called = False
        self.show_post_tensor_transform_called = False
        self.show_collate_called = False
        self.per_batch_transform_called = False

    def show_load_sample(self, samples: List[Any], running_stage: RunningStage):
        self.show_load_sample_called = True

    def show_pre_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        self.show_pre_tensor_transform_called = True

    def show_to_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        self.show_to_tensor_transform_called = True

    def show_post_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        self.show_post_tensor_transform_called = True

    def show_collate(self, batch: Sequence, running_stage: RunningStage) -> None:
        self.show_collate_called = True

    def show_per_batch_transform(self, batch: Sequence, running_stage: RunningStage) -> None:
        self.per_batch_transform_called = True

    def check_reset(self):
        self.show_load_sample_called = False
        self.show_pre_tensor_transform_called = False
        self.show_to_tensor_transform_called = False
        self.show_post_tensor_transform_called = False
        self.show_collate_called = False
        self.per_batch_transform_called = False


@pytest.mark.skipif(not _IMAGE_TESTING, reason="image libraries aren't installed.")
class TestBaseViz:

    @pytest.mark.parametrize(
        "func_names, valid", [
            (["load_sample"], True),
            (["not_a_hook"], False),
            (["load_sample", "pre_tensor_transform"], True),
            (["load_sample", "not_a_hook"], True),
        ]
    )
    def test_show(self, func_names, valid):
        base_viz = CustomBaseVisualization()

        batch = {func_name: "test" for func_name in func_names}

        if not valid:
            with pytest.raises(MisconfigurationException, match="Invalid function names"):
                base_viz.show(batch, RunningStage.TRAINING, func_names)
        else:
            base_viz.show(batch, RunningStage.TRAINING, func_names)
            for func_name in func_names:
                if hasattr(base_viz, f"show_{func_name}_called"):
                    assert getattr(base_viz, f"show_{func_name}_called")
