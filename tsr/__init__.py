"""TSR package.

No eager imports, so CPU only analysis scripts do not load the GPU stack
(torch, transformers, bitsandbytes). Import submodules directly, e.g. `from tsr.agent_runner import AgentRunner`.
"""
