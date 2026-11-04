# A5 — Persona: switch at runtime without losing the conversation.
Output demos/a5-persona.gif
Output demos/a5-persona.mp4
Source demos/_settings.tape

Type "cat config/personas/tutor.yaml"
Enter
Sleep 3s
Type "sensai"
Enter
Wait /^>\s*$/
Type "Je prépare un gâteau pour 6 personnes, la recette est pour 4 et demande 250 g de farine."
Enter
Wait /^>\s*$/
Type "/persona tutor"
Enter
Sleep 1s
Type "Combien de farine me faut-il ?"
Enter
Wait /^>\s*$/
Sleep 2s
Type "Et quel vin servir avec ?"
Enter
Wait /^>\s*$/
Sleep 2s
Type "/persona default"
Enter
Sleep 1s
Type "Et quel vin servir avec ?"
Enter
Wait /^>\s*$/
Sleep 4s
Type "/quit"
Enter
Sleep 1s
