from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    @abstractmethod
    def build_index(self, corpus: List[Dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def requires_gpu(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"