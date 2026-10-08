"""
Inference wrapper for the v3 model saved by scripts/05_train_predictor.py.

It builds features with the same functions as training (tsr/features.py), so
training and serving cannot drift apart.
"""

from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from config import settings
from tsr.features import probe_features, plan_identity, own_plan_features, apply_retriever_stats
from tsr.cv import logit, sigmoid

DEFAULT_PATH = settings.PROJECT_ROOT / "tsr" / "models" / "funnel_predictor.joblib"


class FunnelPredictor:
    def __init__(self, bundle: Dict):
        self.b = bundle
        self.booster = bundle["booster"]
        self.cols = bundle["feature_cols"]
        self.prior = bundle["prior"]
        self.plans = bundle["plans"]
        self.retrievers = bundle["retrievers"]
        self.probe_depth = bundle.get("probe_depth", 100)
        self.cascade = bundle.get("cascade", {})
        self.p_stats = bundle.get("p_stats", {})

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "FunnelPredictor":
        return cls(joblib.load(path or DEFAULT_PATH))

    def _margin_predict(self, booster, rows: List[Dict], cols: List[str], prior: np.ndarray) -> np.ndarray:
        import xgboost as xgb
        X = pd.DataFrame(rows).reindex(columns=cols, fill_value=0.0).to_numpy(np.float32)
        return booster.predict(xgb.DMatrix(X, base_margin=logit(prior)))

    def predict_plans(self, query: str, bm25_probe: List[Dict]) -> pd.DataFrame:
        """Predicted TSR for every plan, from decision time features only."""
        qf = probe_features(query, bm25_probe, self.probe_depth)
        rows, meta = [], []
        for plan in self.plans:
            r, d = plan.rsplit("@", 1)
            rows.append({**qf, **plan_identity(r, int(d), self.retrievers)})
            meta.append((plan, r, int(d)))
        prior = np.array([self.prior[p] for p, _, _ in meta])
        p = self._margin_predict(self.booster, rows, self.cols, prior)
        return pd.DataFrame(meta, columns=["plan", "retriever", "depth"]).assign(p=p, p_prior=prior)

    def predict_cheap_success(self, query: str, bm25_probe: List[Dict], cheap_results: List[Dict]) -> float:
        """P(cheap plan succeeds), using its own retrieved list (cascade step)."""
        c = self.cascade
        if not c or c.get("booster") is None:
            return float(self.prior.get(c.get("cheap"), 0.5))
        r, d = c["cheap"].rsplit("@", 1)
        pf = apply_retriever_stats(own_plan_features(cheap_results, int(d)), r, self.p_stats)
        row = {**probe_features(query, bm25_probe, self.probe_depth), **pf}
        prior = np.array([self.prior.get(c["cheap"], 0.5)])
        return float(self._margin_predict(c["booster"], [row], c["feature_cols"], prior)[0])
