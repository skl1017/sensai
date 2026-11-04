# Sensai — vidéo de démonstration

`sensai-features.mp4` (06:47) enchaîne les 19 démos, attentes raccourcies. Chaque séquence a un chapitre dans la vidéo ; les mêmes timecodes :

| Début | Feature | Ce qu'on voit |
|---|---|---|
| 00:04 | **B1** | Chatbot CLI : streaming, historique de session, erreurs propres, modèle choisi par config |
| 00:29 | **M1** | Sessions persistantes et profil utilisateur injecté à chaque tour |
| 00:52 | **M2** | Budget de tokens : les anciens tours sont résumés, les faits clés restent |
| 01:30 | **M3** | Mémoire structurée SQLite : l'agent enregistre, met à jour, oublie |
| 01:57 | **T1** | MCP : serveur filesystem existant + notre propre serveur (calculatrice) |
| 02:18 | **T2** | Code exécuté en sandbox Docker, l'agent corrige sa propre erreur |
| 02:41 | **T3** | Recherche web (SearXNG) avec sources citées |
| 02:55 | **T4** | Accès fichiers sous permissions : data/ autorisé, le reste refusé |
| 03:15 | **R1** | RAG : documents locaux indexés, réponses avec sources |
| 03:30 | **R2** | RAG avancé : reranking, seuil de score, filtres déduits de la question |
| 03:54 | **A1** | Boucle ReAct : plusieurs appels d'outils enchaînés avant de répondre |
| 04:06 | **A2** | Multi-agents : délégation à un sous-agent tuteur (ask_tutor) |
| 04:21 | **A3** | Human-in-the-loop : refus puis accord avant d'écrire un fichier |
| 04:40 | **A5** | Persona changée en cours de session, historique conservé |
| 05:09 | **A6** | Cache sémantique : même question, réponse servie sans le modèle |
| 05:26 | **A7** | Few-shot choisis selon la question + réécriture des questions de suivi |
| 05:39 | **EV2** | Guardrails : injections bloquées en entrée, PII masquées en sortie |
| 05:58 | **X2** | Branches (édition d'un ancien message) et interruption par Ctrl+C |
| 06:25 | **X4** | Le modèle pose ses questions à l'utilisateur (choix multiple, texte libre) |
