"""
Budget aware retrieval planner (v3).

Change (2026-10-02). The old planner ran EVERY (retriever, depth) that fit
the budget, built features from each result list, and then kept one. So the
real latency was the sum over all candidate plans, not the latency of the
chosen plan; the budget was never respected. It also built features without
the reranker, while the predictor was trained with reranker features, so the
inputs did not match training.

The v3 planner:
  plan():     one BM25 probe (~2 ms) -> predicted TSR for all plans from
              decision time features -> pick the best plan whose expected cost
              (retrieval + agent) fits the budget, or maximise p - lambda*cost
              -> run ONLY that plan.
  cascade():  run the cheap plan, predict whether it will succeed from its own
              result list, and escalate to the strong plan only if not.
"""

import logging
import time
from typing import Any, Dict, Optional

from cost_model import CostModel
from retrievers.base import BaseRetriever
from tsr.predictor import FunnelPredictor

logger = logging.getLogger(__name__)


class Planner:
    def __init__(self, retrievers: Dict[str, BaseRetriever], cost_model: CostModel,
                 predictor: FunnelPredictor, probe_name: str = "bm25"):
        if probe_name not in retrievers:
            raise ValueError(f"The probe retriever '{probe_name}' must be built and passed in.")
        self.retrievers = retrievers
        self.cost_model = cost_model
        self.predictor = predictor
        self.probe_name = probe_name

    def _run(self, retriever: str, depth: int, query: str, probe):
        if retriever == self.probe_name and probe is not None and depth <= len(probe):
            return probe[:depth]
        return self.retrievers[retriever].search(query, depth)

    def plan(self, query: str, budget_s: Optional[float] = None, lam: Optional[float] = None,
             include_agent_cost: bool = True) -> Dict[str, Any]:
        if (budget_s is None) == (lam is None):
            raise ValueError("Give exactly one of budget_s or lam.")
        t0 = time.perf_counter()
        probe = self.retrievers[self.probe_name].search(query, self.predictor.probe_depth)
        preds = self.predictor.predict_plans(query, probe)
        preds = preds[preds["retriever"].isin(self.retrievers)]
        preds["cost"] = [self.cost_model.plan_cost(r, d, include_agent_cost)
                         for r, d in zip(preds["retriever"], preds["depth"])]
        fallback = False
        if budget_s is not None:
            feas = preds[preds["cost"] <= budget_s]
            if feas.empty:
                feas, fallback = preds.nsmallest(1, "cost"), True
            best = feas.loc[feas["p"].idxmax()]
        else:
            best = preds.loc[(preds["p"] - lam * preds["cost"]).idxmax()]
        plan_time = time.perf_counter() - t0
        results = self._run(best["retriever"], int(best["depth"]), query, probe)
        return {
            "retriever": best["retriever"], "depth": int(best["depth"]), "results": results,
            "predicted_tsr": float(best["p"]), "prior_tsr": float(best["p_prior"]),
            "expected_cost_s": float(best["cost"]), "planning_overhead_s": plan_time,
            "fallback_used": fallback, "all_predictions": preds,
        }

    def cascade(self, query: str, tau: float = 0.5) -> Dict[str, Any]:
        c = self.predictor.cascade
        cr, cd = c["cheap"].rsplit("@", 1)
        sr, sd = c["strong"].rsplit("@", 1)
        probe = self.retrievers[self.probe_name].search(query, self.predictor.probe_depth)
        cheap = self._run(cr, int(cd), query, probe)
        p = self.predictor.predict_cheap_success(query, probe, cheap)
        if p >= tau:
            return {"retriever": cr, "depth": int(cd), "results": cheap, "p_cheap": p, "escalated": False}
        return {"retriever": sr, "depth": int(sd), "results": self._run(sr, int(sd), query, probe),
                "p_cheap": p, "escalated": True}
