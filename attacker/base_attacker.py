import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple
import torch


# empty class -- to work with the training loop, it should implement the methods
# that throw errors
class BaseAttacker(nn.Module, ABC):
    def __init__(self):
        super().__init__()

        # need to specify a base model and base attacker
        self.model = None
        self.base_tokenizer = None

    # to be able to use an attacker, should implement the below methods
    @abstractmethod
    def generate_conditional_prompt(
        self, batch_queries: List[str], batch_target_responses: List[str]
    ) -> Tuple[List[str], torch.Tensor]:
        raise NotImplementedError
