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
import urllib.error

import pytest

from flash.image.backbones import catch_url_error


def test_pretrained_backbones_catch_url_error():

    def raise_error_if_pretrained(pretrained=False):
        if pretrained:
            raise urllib.error.URLError('Test error')

    with pytest.warns(UserWarning, match="Failed to download pretrained weights"):
        catch_url_error(raise_error_if_pretrained)(pretrained=True)
