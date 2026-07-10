from typing import List, Tuple
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseLoss(nn.Module, ABC):
    def __init__(self, target_lm, judge, device):
        super().__init__()
        self.target_lm = target_lm
        self.judge = judge
        self.attacker_device = device

    # should implement a forward method, but it returns a dict that contains loss with key 'loss'
    @abstractmethod
    def forward(
        self,
        adversarial_queries: List[str],
        original_queries: List[str],
        log_likelihoods: torch.Tensor,
    ) -> dict:
        raise NotImplementedError
