"""TSR package.

Change (2026-10-02): no eager imports. The old __init__ imported the agent
(torch, transformers, bitsandbytes) on every `import tsr`, so CPU only
analysis scripts such as 05 needed the whole GPU stack. Import submodules
directly, e.g. `from tsr.agent_runner import AgentRunner`.
"""
