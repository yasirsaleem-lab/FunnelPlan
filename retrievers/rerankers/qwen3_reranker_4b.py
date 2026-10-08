from .qwen3_reranker_0_6b import Qwen3Reranker06B
from config import settings


class Qwen3Reranker4B(Qwen3Reranker06B):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("model_name", getattr(settings, "RERANKER_4B", "Qwen/Qwen3-Reranker-4B"))
        kwargs.setdefault("batch_size", 8)
        super().__init__(*args, **kwargs)

    @property
    def name(self) -> str:
        return "qwen3_reranker_4b"
