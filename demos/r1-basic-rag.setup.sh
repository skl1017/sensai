source "$REPO/demos/_lib.sh"
cfg "$CTX['rag'] = {'k': 10, 'min_score': 0.3, 'top_k': 3}"
rag_docs
