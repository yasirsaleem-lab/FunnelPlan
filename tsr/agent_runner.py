"""
Reader (agent) models for TSR labels.

Changes (2026-10-03, EAB plan):
  * READERS registry. One name selects model, precision and generation
    settings, and that name tags every output file. Readers:
        qwen2.5-7b       Qwen/Qwen2.5-7B-Instruct, bf16            (main reader)
        llama3.1-8b      meta-llama/Llama-3.1-8B-Instruct, bf16    (second family)
        qwen3-14b        Qwen/Qwen3-14B, int8, thinking disabled   (strong reader)
        qwen2.5-7b-4bit  the old v3 setting: nf4, 32 new tokens    (legacy, for comparison)
    bf16 and a 100 token limit follow "The Powerless Noise" (SIGIR 2026),
    which showed 4 bit quantisation and short output limits change RAG
    conclusions. The 14B reader needs int8 to fit 20 GB with long prompts;
    int8 is reported near lossless (Dettmers et al., NeurIPS 2022).
  * Depth 0 (no passages) uses a closed book prompt, to measure what the
    reader already knows.
  * Answer token log probabilities are returned (mean, min, first token) for
    the confidence analysis (RQ4).
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import settings
from data.corpus_utils import build_text_by_id, passage_text

logger = logging.getLogger(__name__)
# bitsandbytes int8 logs "inputs will be cast from bfloat16 to float16" on every
# matmul. The cast is expected (LLM.int8 computes in fp16) and harmless; the
# flood hides the progress bar, so silence that one logger.
logging.getLogger("bitsandbytes.autograd._functions").setLevel(logging.ERROR)

MAX_PASSAGES_TO_PROMPT = 100

READERS: Dict[str, Dict[str, Any]] = {
    "qwen2.5-7b":      {"hf": "Qwen/Qwen2.5-7B-Instruct", "quant": None, "max_new_tokens": 100},
    # Llama is gated on Hugging Face; download it from ModelScope into "local".
    "llama3.1-8b":     {"hf": "meta-llama/Llama-3.1-8B-Instruct", "quant": None, "max_new_tokens": 100,
                        "local": r"C:\Users\USER\models\Meta-Llama-3.1-8B-Instruct"},
    "qwen3-14b":       {"hf": "Qwen/Qwen3-14B", "quant": "int8", "max_new_tokens": 100,
                        "chat_kwargs": {"enable_thinking": False}},
    "qwen2.5-7b-4bit": {"hf": "Qwen/Qwen2.5-7B-Instruct", "quant": "nf4", "max_new_tokens": 32},
}

SYSTEM_RAG = ("You answer questions using only the given passages. "
              "Reply with the short answer only (a few words). "
              "If the passages do not contain the answer, give your best guess.")
SYSTEM_CLOSED = ("Answer the question from your own knowledge. "
                 "Reply with the short answer only (a few words).")


class AgentRunner:
    def __init__(self, reader: str = "qwen2.5-7b", device: Optional[str] = None,
                 model_name: Optional[str] = None, use_4bit: Optional[bool] = None,
                 use_chat_template: bool = True):
        if reader not in READERS:
            raise ValueError(f"Unknown reader {reader}. Choose from {list(READERS)}")
        self.reader = reader
        self.spec = dict(READERS[reader])
        local = self.spec.get("local")
        if local and os.path.isdir(local):
            self.spec["hf"] = local          # load from the local folder, not the hub
        if model_name:
            self.spec["hf"] = model_name
        if use_4bit is True:          # backward compatibility with old callers
            self.spec["quant"] = "nf4"
        self.model_name = self.spec["hf"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_chat_template = use_chat_template
        self.tokenizer = None
        self.model = None
        self._loaded = False
        self.corpus_text_by_id: Optional[Dict[str, str]] = None

    def set_corpus(self, corpus: Dict[str, Dict[str, Any]]) -> None:
        self.corpus_text_by_id = build_text_by_id(corpus)
        logger.info(f"Agent corpus set: {len(self.corpus_text_by_id)} documents")

    def load(self) -> None:
        if self._loaded:
            return
        q = self.spec["quant"]
        logger.info(f"Loading reader {self.reader} ({self.model_name}, quant={q or 'bf16'}) on {self.device}")
        qcfg = None
        if self.device.startswith("cuda") and q == "nf4":
            qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                      bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        elif self.device.startswith("cuda") and q == "int8":
            qcfg = BitsAndBytesConfig(load_in_8bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, quantization_config=qcfg,
            device_map={"": 0} if self.device.startswith("cuda") else None,
            torch_dtype=torch.bfloat16, trust_remote_code=True,   # torch_dtype works on old and new transformers
        )
        self.model.eval()
        self._loaded = True
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            logger.info(f"VRAM after reader load: free {free / 1024**3:.1f} GB / {total / 1024**3:.1f} GB")

    def unload(self) -> None:
        if not self._loaded:
            return
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False

    def _resolve(self, p: Dict[str, Any]) -> str:
        if p.get("text"):
            return passage_text(p.get("title", ""), p["text"])
        if self.corpus_text_by_id is None or p.get("id") is None:
            return ""
        return self.corpus_text_by_id.get(str(p["id"]), "")

    def build_prompt(self, query: str, passages: List[Dict[str, Any]]) -> str:
        if passages:
            lines = [f"Passage {i + 1}: {t}" for i, p in enumerate(passages[:MAX_PASSAGES_TO_PROMPT])
                     if (t := self._resolve(p))]
            user = "Passages:\n" + "\n".join(lines) + f"\n\nQuestion: {query}\nAnswer:"
            system = SYSTEM_RAG
        else:
            user = f"Question: {query}\nAnswer:"
            system = SYSTEM_CLOSED
        if not self.use_chat_template:
            return f"{system}\n{user}"
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                  **self.spec.get("chat_kwargs", {}))

    def answer_with_meta(self, query: str, passages: List[Dict[str, Any]],
                         max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        if not self._loaded:
            self.load()
        prompt = self.build_prompt(query, passages)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        n_prompt = int(inputs.input_ids.shape[1])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens or self.spec["max_new_tokens"],
                do_sample=False, temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tokenizer.pad_token_id,
                output_scores=True, return_dict_in_generate=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gen = out.sequences[0][n_prompt:]
        lp = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
        lp = lp[: len(gen)].float()
        keep = [i for i, t in enumerate(gen.tolist()) if t not in self.tokenizer.all_special_ids]
        lpk = lp[keep] if keep else lp
        return {
            "answer": self.tokenizer.decode(gen, skip_special_tokens=True).strip(),
            "prompt_tokens": n_prompt,
            "gen_tokens": int(gen.shape[0]),
            "agent_time_s": dt,
            "mean_logprob": float(lpk.mean()) if lpk.numel() else float("nan"),
            "min_logprob": float(lpk.min()) if lpk.numel() else float("nan"),
            "first_token_prob": float(lpk[0].exp()) if lpk.numel() else float("nan"),
        }

    def answer(self, query: str, passages: List[Dict[str, Any]], max_new_tokens: Optional[int] = None,
               temperature: float = 0.0) -> str:
        return self.answer_with_meta(query, passages, max_new_tokens)["answer"]
