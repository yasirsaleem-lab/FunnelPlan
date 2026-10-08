"""Tests for the v3 cost model (retrieval + agent)."""

import json
import tempfile
from pathlib import Path

import numpy as np

from cost_model import CostModel
from cost_model.fit_cost_model import fit_line_nonneg


def test_plan_cost_adds_agent():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "params.json"
        p.write_text(json.dumps({"retrieval": {"bm25": {"c": 0.002, "a": 0.0}},
                                 "agent": {"a0": 0.3, "a1": 1e-4, "t0": 100, "tokens_per_passage": 150}}))
        cm = CostModel(params_file=p)
        assert np.isclose(cm.retrieval_latency("bm25", 100), 0.002)
        assert np.isclose(cm.agent_latency(100), 0.3 + 1e-4 * (100 + 150 * 100))
        assert cm.plan_cost("bm25", 100) > cm.plan_cost("bm25", 10)


def test_legacy_format_still_reads():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "params.json"
        p.write_text(json.dumps({"bm25": {"a": 0.001, "b": 1.0, "c": 0.001}}))
        cm = CostModel(params_file=p)
        assert np.isclose(cm.retrieval_latency("bm25", 100), 0.101)


def test_nonneg_line_fit():
    x = np.array([10, 50, 100, 200.0])
    c, a = fit_line_nonneg(x, 0.2 + 0.001 * x)
    assert np.isclose(c, 0.2, atol=1e-6) and np.isclose(a, 0.001, atol=1e-8)
