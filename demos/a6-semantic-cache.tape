# A6 — Semantic cache: the same question in a new session skips the model.
Output demos/a6-semantic-cache.gif
Output demos/a6-semantic-cache.mp4
Source demos/_settings.tape
Hide
Type "export SENSAI_SHOW_STATE=cache_hit && clear"
Enter
Show

Type "sensai"
Enter
Wait /^>\s*$/
Type "Qu'est-ce que la photosynthèse ? Réponds en trois phrases."
Enter
Wait /^>\s*$/
Sleep 2s
Type "/quit"
Enter
Sleep 1s
Type "clear"
Enter

Type "sensai"
Enter
Wait /^>\s*$/
Type "Qu'est-ce que la photosynthèse ? Réponds en trois phrases."
Enter
Wait /^>\s*$/
Sleep 4s
Type "/quit"
Enter
Sleep 1s
