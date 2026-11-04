# X4 — Questions: the model asks the user mid-turn instead of guessing.
Output demos/x4-ask-user.gif
Output demos/x4-ask-user.mp4
Source demos/_settings.tape

Type "sensai"
Enter
Wait /^>\s*$/
Type "Aide-moi à choisir un cadeau. Demande-moi d'abord pour qui c'est (choix multiple), puis ses passions (question libre)."
Enter
Wait /Choice \[1-\d+\]:\s*$/
Sleep 2s
Type "2"
Enter
Wait /[?):]\s*$/
Sleep 1s
Type "Randonnée et photo, budget 50 euros."
Enter
Wait /^>\s*$/
Sleep 4s
Type "/quit"
Enter
Sleep 1s
