# A3 — Human-in-the-loop: a side-effecting tool waits for an explicit yes.
Output demos/a3-human-in-the-loop.gif
Output demos/a3-human-in-the-loop.mp4
Source demos/_settings.tape

Type "grep -A4 '^approval' config/agent.yaml"
Enter
Sleep 3s
Type "sensai"
Enter
Wait /^>\s*$/
Type "Crée le fichier data/todo.md avec trois tâches pour préparer un déménagement."
Enter
Wait /no\]\s*$/
Sleep 2s
# Refused: nothing is written.
Type "no"
Enter
Wait /^>\s*$/
Sleep 2s
Type "Réessaie, cette fois je suis d'accord."
Enter
Wait /no\]\s*$/
Sleep 2s
Type "yes"
Enter
Wait /^>\s*$/
Sleep 1s
Type "/quit"
Enter
Sleep 1s
Type "cat data/todo.md"
Enter
Sleep 4s
