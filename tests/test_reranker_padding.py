"""Regression test for the reranker padding bug (needs torch + transformers
installed, but no model download)."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from retrievers.rerankers.qwen3_reranker_0_6b import left_pad


def test_left_pad_puts_real_tokens_last():
    ids, mask = left_pad([[5, 6, 7], [8]], pad_id=0)
    assert ids == [[5, 6, 7], [0, 0, 8]]
    assert mask == [[1, 1, 1], [0, 0, 1]]
    # The scorer reads position -1. It must be a real token for every row.
    assert all(row[-1] != 0 for row in ids)
    assert all(m[-1] == 1 for m in mask)
