"""
Plan cost model: retrieval latency + agent latency.

    cost(r, d) = retrieval(r, d) + agent(d)
    retrieval(r, d) = c_r + a_r * d           (linear; profiles are near flat in d)
    agent(d)        = a0 + a1 * prompt_tokens(d) + a2 * gen_mean   (a2, gen_mean: full mode)
    prompt_tokens(d) = t0 + tokens_per_passage * d

Parameters live in data/processed/cost_model_params.json:
    {"retrieval": {name: {"c":..,"a":..}}, "agent": {"a0":..,"a1":..,"t0":..,"tokens_per_passage":..},
     "source": {...}}
The old flat power law format {name: {"a","b","c"}} is still read.
"""

import json
from pathlib import Path
from typing import Dict, Optional

from config import settings

# Engineering estimates, NOT measurements. They are used only until
# fit_cost_model.py has fitted agent parameters from tsr_labels_v3.csv.
DEFAULT_AGENT = {
    "a0": 0.30,                 # s, fixed decode + overhead for a short answer
    "a1": 1.2e-4,               # s per prompt token (prefill, 4 bit 7B)
    "t0": 120.0,                # tokens of instructions + question
    "tokens_per_passage": 150.0,
    "estimated": True,
}


class CostModel:
    PARAMS_FILE = settings.PROCESSED_DATA_ROOT / "cost_model_params.json"

    def __init__(self, params_file: Optional[Path] = None, dataset: Optional[str] = None,
                 reader: Optional[str] = None):
        """dataset/reader pick the full mode per dataset fit ("by_dataset").
        Without them, or for a pilot file, the pooled fit is used."""
        self.params_file = Path(params_file or self.PARAMS_FILE)
        self.retrieval: Dict[str, Dict[str, float]] = {}
        self.agent: Dict[str, float] = dict(DEFAULT_AGENT)
        self.source: Dict = {}
        self.by_dataset: Dict = {}
        if self.params_file.exists():
            raw = json.loads(self.params_file.read_text())
            self.by_dataset = raw.get("by_dataset", {})
            if dataset is not None and dataset in self.by_dataset:
                e = self.by_dataset[dataset]
                self.retrieval = e.get("retrieval", {})
                agents = e.get("agent", {})
                pick = reader if reader in agents else (next(iter(agents)) if agents else None)
                if reader is not None and reader not in agents:
                    raise KeyError(f"No agent cost for reader '{reader}' on {dataset}; have {list(agents)}")
                if pick:
                    self.agent.update(agents[pick])
                self.source = raw.get("source", {})
            elif dataset is not None and self.by_dataset:
                raise KeyError(f"No cost model for dataset '{dataset}'; have {list(self.by_dataset)}")
            elif "retrieval" in raw:
                self.retrieval = raw["retrieval"]
                self.agent.update(raw.get("agent", {}))
                self.source = raw.get("source", {})
            else:  # legacy power law format
                self.retrieval = {k: {"legacy": v} for k, v in raw.items()}

    # ------------------------------------------------------------------
    def retrieval_latency(self, retriever: str, depth: int) -> float:
        p = self.retrieval.get(retriever)
        if p is None:
            raise KeyError(f"No cost parameters for retriever '{retriever}'. Run fit_cost_model.")
        if "legacy" in p:
            q = p["legacy"]
            return q["a"] * depth ** q["b"] + q["c"]
        return p["c"] + p["a"] * depth

    def prompt_tokens(self, depth: int, retriever: Optional[str] = None) -> float:
        per = self.agent.get("prompt_by_retriever", {}).get(retriever) if retriever else None
        if per and depth > 0:
            return per["t0"] + per["tokens_per_passage"] * depth
        return self.agent["t0"] + self.agent["tokens_per_passage"] * depth

    def agent_latency(self, depth: int, prompt_tokens: Optional[float] = None,
                      retriever: Optional[str] = None) -> float:
        n = self.prompt_tokens(depth, retriever) if prompt_tokens is None else prompt_tokens
        # decode term (full mode fits); 0 for older parameter files
        decode = self.agent.get("a2", 0.0) * self.agent.get("gen_mean", 0.0)
        return self.agent["a0"] + self.agent["a1"] * n + decode

    def plan_cost(self, retriever: str, depth: int, include_agent: bool = True) -> float:
        t = self.retrieval_latency(retriever, depth)
        return t + (self.agent_latency(depth, retriever=retriever) if include_agent else 0.0)

    # Backward compatible name used by older code.
    def predict_latency(self, retriever_name: str, k: int) -> float:
        return self.retrieval_latency(retriever_name, k)

    def save(self) -> None:
        self.params_file.parent.mkdir(parents=True, exist_ok=True)
        self.params_file.write_text(json.dumps(
            {"retrieval": self.retrieval, "agent": self.agent, "source": self.source,
             "by_dataset": self.by_dataset}, indent=2))

    @classmethod
    def load(cls) -> "CostModel":
        return cls()
