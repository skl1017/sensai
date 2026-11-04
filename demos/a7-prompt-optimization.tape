# A7 — Prompt optimization: closest few-shot examples and query rewrite.
Output demos/a7-prompt-optimization.gif
Output demos/a7-prompt-optimization.mp4
Source demos/_settings.tape
Hide
Type "export SENSAI_SHOW_STATE=query,fewshot && clear"
Enter
Show

Type "grep -c 'question:' config/fewshot.yaml"
Enter
Sleep 2s
Type "sensai"
Enter
Wait /^>\s*$/
# Few-shot: the examples closest to the question are injected.
Type "Comment simplifier la fraction 45/60 ?"
Enter
Wait /^>\s*$/
Sleep 3s
Type "Parle-moi de la tour Eiffel en deux phrases."
Enter
Wait /^>\s*$/
# Query rewrite: the follow-up is made self-contained.
Type "Elle mesure combien ?"
Enter
Wait /^>\s*$/
Sleep 4s
Type "/quit"
Enter
Sleep 1s
