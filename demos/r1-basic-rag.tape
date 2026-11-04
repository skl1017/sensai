# R1 — RAG: local documents, chunked, embedded, retrieved and cited.
Output demos/r1-basic-rag.gif
Output demos/r1-basic-rag.mp4
Source demos/_settings.tape

Type "ls .docs/"
Enter
Sleep 2s
Type "sensai"
Enter
Wait /^>\s*$/
Type "Combien de jours de télétravail par semaine ai-je droit, et quel jour dois-je venir ?"
Enter
Wait /^>\s*$/
Sleep 3s
Type "Quelle est l'indemnité de télétravail, et est-elle plafonnée ?"
Enter
Wait /^>\s*$/
Sleep 4s
Type "/quit"
Enter
Sleep 1s
