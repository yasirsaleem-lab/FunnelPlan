"""
Pytest fixtures shared across test modules.
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from retrievers.base import BaseRetriever


class MockRetriever(BaseRetriever):
    """A simple mock retriever that returns predefined scores for testing."""

    def __init__(self, name: str = "mock", scores: List[float] = None):
        self.name_ = name
        self.scores = scores or [0.5, 0.4, 0.3, 0.2, 0.1]
        self.corpus_ids = [f"doc_{i}" for i in range(len(self.scores))]

    def build_index(self, corpus: List[Dict[str, Any]]) -> None:
        # mock: do nothing
        pass

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        # Return top-k scores
        top_k = min(k, len(self.scores))
        results = []
        for i in range(top_k):
            results.append({
                "id": self.corpus_ids[i],
                "score": self.scores[i],
            })
        return results

    @property
    def name(self) -> str:
        return self.name_

    @property
    def requires_gpu(self) -> bool:
        return False


@pytest.fixture
def mock_retriever():
    return MockRetriever(name="mock")


@pytest.fixture
def mock_corpus():
    """Return a small list of documents."""
    return [{"id": f"doc_{i}", "text": f"text {i}"} for i in range(10)]


@pytest.fixture
def mock_queries():
    return ["query1", "query2", "query3"]


@pytest.fixture
def mock_cost_model_params():
    """Return dummy parameters for cost model."""
    return {
        "bm25": {"a": 0.001, "b": 1.0, "c": 0.001},
        "dense": {"a": 0.005, "b": 1.2, "c": 0.002},
    }


@pytest.fixture
def mock_tsr_predictor():
    """Mock TSR predictor that returns constant probability."""
    class MockPredictor:
        def predict_proba(self, features):
            return 0.8
        def predict(self, features):
            return True
    return MockPredictor()


@pytest.fixture
def mock_planner_dependencies(mock_retriever, mock_tsr_predictor):
    """Return a dict of retrievers, cost model, etc. for planner tests."""
    from cost_model import CostModel
    cm = CostModel()
    cm.update_params("mock", 0.001, 1.0, 0.001)
    retrievers = {"mock": mock_retriever}
    return {
        "retrievers": retrievers,
        "cost_model": cm,
        "tsr_predictor": mock_tsr_predictor,
        "fallback_name": "mock",
    }