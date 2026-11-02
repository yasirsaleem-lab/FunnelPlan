Files from the code release of Moskvoretskii et al., "Adaptive Retrieval Without Self-Knowledge? Bringing Uncertainty
Back Home" (ACL 2025), https://github.com/s-nlp/AdaRAGUE (MIT licence; the Adaptive_Rag folder is Apache 2.0).
  test_<dataset>.csv                      their 500 test questions per dataset, gold references, and their Llama 3.1 8B
                                          answers without retrieval (no_retrieve) and with one retrieval (rag_one_retrieve)
                                          (columns dataset, qid, question_text, reference, no_retrieve, rag_one_retrieve)
  no_context_cot_qa_codex_<dataset>.txt   Adaptive_Rag/prompts/<dataset>/no_context_cot_qa_codex.txt, the few shot
                                          prompt of their no retrieval runs (base_configs/nor_qa_llama_8b_it_<dataset>.jsonnet)
  valid_qids_nor_qa_llama_8b_it.json      the demonstration ids those configs select, in their order
Used by scripts/31_closed_book_prompt.py.
