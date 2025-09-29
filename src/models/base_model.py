from abc import ABC, abstractmethod
from typing import Any, List

class Recommender(ABC):
    """Abstract Base Class for all recommendation system models."""
    @abstractmethod
    def train(self, data: Any, **kwargs) -> None:
        pass

    @abstractmethod
    def recommend(self, user_id: int, top_k: int) -> List[str]:
        pass
    
    def format(self, recommendations: List[str]) -> str:
        return ','.join(map(str, recommendations))