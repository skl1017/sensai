# M3 — Structured memory (SQLite): the agent saves, reads, updates and forgets facts.
Output demos/m3-structured-memory.gif
Output demos/m3-structured-memory.mp4
Source demos/_settings.tape

Type "sensai"
Enter
Wait /^>\s*$/
Type "Retiens que mon chat s'appelle Miso, qu'il a 3 ans et qu'il est allergique au poulet."
Enter
Wait /^>\s*$/
Type "/memory list"
Enter
Sleep 3s
Type "/quit"
Enter
Sleep 1s
Type "clear"
Enter

# New session: the facts come back from the database.
Type "sensai"
Enter
Wait /^>\s*$/
Type "Quelle viande dois-je éviter dans les croquettes de mon chat ?"
Enter
Wait /^>\s*$/
Sleep 2s
Type "Miso vient d'avoir 4 ans, mets à jour."
Enter
Wait /^>\s*$/
Type "/memory list"
Enter
Sleep 3s
# Forgetting is irreversible: it needs approval.
Type "Oublie l'âge de Miso."
Enter
Wait /no\]\s*$/
Sleep 1s
Type "yes"
Enter
Wait /^>\s*$/
Type "/memory list"
Enter
Sleep 3s
Type "/quit"
Enter
Sleep 1s
