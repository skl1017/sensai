# M1 — Persistent sessions and user profile, across two separate runs.
Output demos/m1-sessions-profile.gif
Output demos/m1-sessions-profile.mp4
Source demos/_settings.tape

Type "sensai --session cours-japonais"
Enter
Wait /^>\s*$/

# A persistent profile, injected into every turn.
Type "/profile set niveau débutant"
Enter
Type "/profile instructions Termine chaque réponse par un mot japonais et sa traduction."
Enter
Type "/profile"
Enter
Sleep 2s

Type "Comment saluer quelqu'un le matin en japonais ?"
Enter
Wait /^>\s*$/
Sleep 2s
Type "/quit"
Enter
Sleep 1s
Type "clear"
Enter

# A brand-new process: the session is on disk.
Type "sensai"
Enter
Wait /^>\s*$/
Type "/sessions"
Enter
Sleep 2s
Type "/resume cours-japonais"
Enter
Sleep 2s
Type "Et le soir ?"
Enter
Wait /^>\s*$/
Sleep 3s
Type "/quit"
Enter
Sleep 1s
