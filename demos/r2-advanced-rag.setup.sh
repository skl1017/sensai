source "$REPO/demos/_lib.sh"
cfg "$CTX['rag'] = {'k': 20, 'min_score': 0.3, 'rerank': True, 'rerank_min_score': 0.1, 'top_k': 4, 'extract_filters': True}"
rag_docs
