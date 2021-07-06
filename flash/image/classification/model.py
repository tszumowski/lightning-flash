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
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Type, Union

import torch
from torchmetrics import Metric

from flash.core.classification import ClassificationTask, Labels
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.process import Serializer


class ImageClassifier(ClassificationTask):

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        loss_fn: Optional[Callable] = None,
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        metrics: Union[Metric, Callable, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-3,
        serializer: Optional[Union[Serializer, Mapping[str, Serializer]]] = None,
    ):
        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            metrics=metrics,
            learning_rate=learning_rate,
            serializer=serializer or Labels(),
        )

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (
            batch[DefaultDataKeys.INPUT],
            batch[DefaultDataKeys.TARGET],
        )
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().validation_step(batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        batch = (batch[DefaultDataKeys.INPUT], batch[DefaultDataKeys.TARGET])
        return super().test_step(batch, batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        batch[DefaultDataKeys.PREDS] = super().predict_step(
            (batch[DefaultDataKeys.INPUT]),
            batch_idx,
            dataloader_idx=dataloader_idx,
        )
        return batch
