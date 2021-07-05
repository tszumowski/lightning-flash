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
import pytest
from PIL import Image

from flash.core.data.data_source import DefaultDataKeys
from flash.image.classification.data import ImageClassificationPathsDataSource


@pytest.fixture
def image_classification_tmpdir(tmpdir):
    (tmpdir / "ants").mkdir()
    (tmpdir / "bees").mkdir()
    Image.new('RGB', (128, 128)).save(str(tmpdir / "ants" / "ant.png"))
    Image.new('RGB', (128, 128)).save(str(tmpdir / "bees" / "bee.png"))
    return tmpdir


def test_image_classification_paths_data_source(image_classification_tmpdir):
    data_source = ImageClassificationPathsDataSource()
    samples = data_source.load_data(image_classification_tmpdir)
    assert len(samples) == 2

    classes = set()
    for sample in samples:
        sample = data_source.load_sample(sample)
        assert sample[DefaultDataKeys.INPUT].size == (128, 128)
        classes.add(sample[DefaultDataKeys.TARGET])

    assert classes == {0, 1}
