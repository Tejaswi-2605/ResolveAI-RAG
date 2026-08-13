"""
app.core — the agent and its safety machinery.

    security.py    the trust boundary: fencing, injection detection, redaction
    validation.py  the structured-output contract
    prompts.py     versioned system prompts and the three-region message layout
    tools.py       the only actions the agent may take
    agent.py       the bounded orchestration loop

Nothing in this package imports faiss, an embedding model, BM25 or the
reranker. Knowledge reaches the agent only through `tools.search_knowledge_base`.
"""
