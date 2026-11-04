# M2 — Token budget: past the threshold, old turns are summarized (tiny window on purpose).
Output demos/m2-token-budget.gif
Output demos/m2-token-budget.mp4
Source demos/_settings.tape
Hide
Type "export SENSAI_SHOW_STATE=context_stats,summary && clear"
Enter
Show

Type "grep -n -A1 'max_context_tokens\|budget' config/agent.yaml"
Enter
Sleep 3s
Type "clear"
Enter

Type "sensai"
Enter
Wait /^>\s*$/
Type "Je m'appelle Alex, je suis allergique aux arachides et je pars à Kyoto en avril."
Enter
Wait /^>\s*$/
Type "Décris le quartier de Gion en un paragraphe."
Enter
Wait /^>\s*$/
Type "Décris le sanctuaire Fushimi Inari en un paragraphe."
Enter
Wait /^>\s*$/
Type "Décris le Pavillon d'or en un paragraphe."
Enter
Wait /^>\s*$/
Sleep 2s
# The first turn is now only in the summary.
Type "Rappelle-moi mon allergie et mon mois de départ."
Enter
Wait /^>\s*$/
Sleep 4s
Type "/quit"
Enter
Sleep 1s
