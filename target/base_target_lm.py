from abc import ABC, abstractmethod
from typing import List

# Base Target LM Wrapper
# Only needs to implement one method -- batch_respond, which takes queries


class BaseTarget(ABC):
    def __init__(self):
        pass

    def respond_to_query(self, query: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def batch_respond(self, queries: List[str]) -> List[str]:
        raise NotImplementedError
    
    @abstractmethod
    def get_generation_pipeline(self):
        raise NotImplementedError
