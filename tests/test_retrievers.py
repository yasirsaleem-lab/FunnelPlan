"""
Unit tests for retrievers: interface conformance and basic search.
"""

import pytest
from retrievers.base import BaseRetriever
from retrievers.sparse.bm25 import BM25Retriever
from retrievers.dense.stella_1_5b import StellaRetriever
from retrievers.hybrid.gte_bm25_rrf import HybridRRFRetriever

# Only the base interface and BM25 are tested; dense models are too heavy for unit tests.

def test_base_interface():
    """Ensure BaseRetriever has abstract methods."""
    with pytest.raises(TypeError):
        # an abstract class cannot be instantiated
        BaseRetriever()


def test_bm25_build_and_search(mock_corpus):
    """Test BM25 index building and search returns correct format."""
    ret = BM25Retriever()
    ret.build_index(mock_corpus)
    results = ret.search("test query", k=3)
    assert len(results) == 3
    for r in results:
        assert "id" in r
        assert "score" in r
        assert isinstance(r["score"], float)


def test_hybrid_retriever_build_and_search(mock_corpus):
    """Test HybridRRF builds and searches without errors."""
    ret = HybridRRFRetriever()
    ret.build_index(mock_corpus)
    results = ret.search("test", k=2)
    assert len(results) == 2
    for r in results:
        assert "id" in r
        assert "score" in r


def test_retriever_name_property(mock_retriever):
    """Test that each retriever has a name."""
    assert mock_retriever.name == "mock"
    bm = BM25Retriever()
    assert bm.name == "bm25"