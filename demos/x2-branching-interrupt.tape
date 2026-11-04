# X2 — Branching (edit a past message into a new branch) and interrupt (Ctrl+C mid-stream).
Output demos/x2-branching-interrupt.gif
Output demos/x2-branching-interrupt.mp4
Source demos/_settings.tape

Type "sensai --session voyage"
Enter
Wait /^>\s*$/
Type "/history"
Enter
Sleep 3s
# Fork: rewrite the second question, the original branch is kept.
Type "/edit u2000000 Donne-moi plutôt un programme pour une journée de pluie."
Enter
Wait /^>\s*$/
Sleep 2s
Type "/tree"
Enter
Sleep 4s
# Interrupt a long answer and redirect it.
Type "Écris un très long récit de voyage à Lisbonne, au moins 1500 mots."
Enter
Sleep 6s
Ctrl+C
Wait /^>\s*$/
Sleep 1s
Type "Trop long : résume plutôt la journée en un haïku."
Enter
Wait /^>\s*$/
Sleep 2s
Type "/tree"
Enter
Sleep 5s
Type "/quit"
Enter
Sleep 1s
