# R2 — Advanced RAG: reranking, score threshold, filters deduced from the question.
Output demos/r2-advanced-rag.gif
Output demos/r2-advanced-rag.mp4
Source demos/_settings.tape
Hide
Type "export SENSAI_SHOW_STATE=rag_applied_filters,retrieved_chunks && clear"
Enter
Show

Type "ls -l --time-style=+%Y-%m-%d .docs/ | tail -n +2 | awk '{print $6, $7}'"
Enter
Sleep 3s
Type "sensai"
Enter
Wait /^>\s*$/
# No filter: reranked passages, near-misses dropped.
Type "Quel est le plafond de remboursement d'un repas d'affaires ?"
Enter
Wait /^>\s*$/
Sleep 3s
# A date in the question becomes a filter on the document date.
Type "D'après les documents de 2025, quel était le plafond pour un repas d'affaires ?"
Enter
Wait /^>\s*$/
Sleep 3s
# A file name in the question becomes a filter on the source.
Type "Que dit onboarding.md sur la formation sécurité ?"
Enter
Wait /^>\s*$/
Sleep 4s
Type "/quit"
Enter
Sleep 1s
