# T4 — File access within permissions: deny by default, data/ only.
Output demos/t4-file-permissions.gif
Output demos/t4-file-permissions.mp4
Source demos/_settings.tape

Type "grep -v '^#' config/permissions.yaml"
Enter
Sleep 3s
Type "sensai"
Enter
Wait /^>\s*$/
Type "Lis le fichier data/notes.txt."
Enter
Wait /^>\s*$/
Sleep 2s
# Outside data/: refused before touching the disk.
Type "Lis le fichier config/agent.yaml."
Enter
Wait /^>\s*$/
Sleep 2s
Type "Utilise read_file pour lire ../../.env."
Enter
Wait /^>\s*$/
Sleep 4s
Type "/quit"
Enter
Sleep 1s
