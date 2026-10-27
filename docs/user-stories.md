# User stories — Sensai

Source : issues de [`skl1017/sensai`](https://github.com/skl1017/sensai/issues), état au 27/10/2026. Seules les features terminées à au moins 50 % sont listées. Chaque feature couvre au moins deux user stories, chacune avec ses scénarios d'acceptation.

**Rôles** : *utilisateur* (discute avec l'agent), *opérateur* (configure et déploie), *développeur* (étend Sensai).

| Feature | Issues | Avancement | US |
|---|---|---|---|
| F1 Socle et extensibilité | [#1](https://github.com/skl1017/sensai/issues/1), [#7](https://github.com/skl1017/sensai/issues/7), [#18](https://github.com/skl1017/sensai/issues/18) | 12/12 | 3 |
| F2 Conversation avec outils | [#12](https://github.com/skl1017/sensai/issues/12) | 5/5 | 3 |
| F3 Sessions et branches | [#22](https://github.com/skl1017/sensai/issues/22) | 4/4 | 3 |
| F4 Personnalisation et contexte | [#27](https://github.com/skl1017/sensai/issues/27) | 5/5 | 3 |
| F5 Contrôle humain | [#33](https://github.com/skl1017/sensai/issues/33) | 4/4 | 3 |
| F6 Outils système | [#38](https://github.com/skl1017/sensai/issues/38) | 3/4 (reste [#42](https://github.com/skl1017/sensai/issues/42)) | 3 |
| F7 MCP | [#43](https://github.com/skl1017/sensai/issues/43) | 4/4 | 2 |
| F8 RAG | [#53](https://github.com/skl1017/sensai/issues/53) | 4/5 (reste [#97](https://github.com/skl1017/sensai/issues/97)) | 3 |
| F9 Multi-agents | [#77](https://github.com/skl1017/sensai/issues/77) | 2/3 (reste [#80](https://github.com/skl1017/sensai/issues/80)) | 2 |

F1 regroupe les stories techniques 1 (cœur), 2 (Ollama) et 4 (configuration) : prises séparément, elles ne portent pas chacune deux besoins distincts.

---

## F1 — Socle et extensibilité ([#1](https://github.com/skl1017/sensai/issues/1), [#7](https://github.com/skl1017/sensai/issues/7), [#18](https://github.com/skl1017/sensai/issues/18))

**US 1.1** — En tant que développeur, je veux ajouter un node ou un outil sans modifier le cœur, afin de faire évoluer l'agent sans risque de régression.

```gherkin
Scénario: Ajout d'un node
  Étant donné un nouveau node enregistré dans NODES
  Quand j'ajoute sa ligne dans la pipeline de agent.yaml
  Alors il s'exécute à cette position, sans changement dans core/
```

**US 1.2** — En tant que développeur, je veux tester l'agent sans lancer Ollama, afin d'avoir une suite de tests rapide et reproductible.

```gherkin
Scénario: Test avec FakeLLM
  Étant donné un FakeLLM scripté avec un appel d'outil puis une réponse
  Quand j'exécute l'agent
  Alors le test passe sans aucun appel réseau
```

**US 1.3** — En tant qu'opérateur, je veux configurer l'agent par un fichier YAML avec des erreurs explicites, afin d'activer ou retirer une fonctionnalité sans toucher au code.

```gherkin
Scénario: Retrait d'une fonctionnalité
  Quand je supprime la ligne "persist" de la pipeline
  Alors l'agent se construit sans PersistNode

Scénario: Modèle absent
  Quand le modèle configuré n'est pas installé dans Ollama
  Alors l'erreur invite à lancer "ollama pull <model>"
```

## F2 — Conversation avec outils ([#12](https://github.com/skl1017/sensai/issues/12))

**US 2.1** — En tant qu'utilisateur, je veux discuter avec l'agent dans mon terminal en voyant la réponse s'écrire, afin de ne pas attendre la fin de la génération.

```gherkin
Scénario: Streaming en CLI
  Quand j'envoie un message dans la CLI
  Alors la réponse s'affiche au fil de l'eau
```

**US 2.2** — En tant qu'utilisateur, je veux que l'agent utilise des outils quand c'est utile, afin d'obtenir des résultats exacts.

```gherkin
Scénario: Calcul
  Quand je demande "combien font 12 fois 7 ?"
  Alors le modèle appelle calculator et répond 84
```

**US 2.3** — En tant qu'utilisateur, je veux que l'agent se remette d'une erreur d'outil, afin de toujours obtenir une réponse.

```gherkin
Scénario: Outil en erreur
  Quand un outil lève une exception ou dépasse son timeout
  Alors l'erreur revient au modèle comme observation et la requête aboutit

Scénario: Boucle infinie évitée
  Quand le modèle répète 3 fois le même appel
  Alors un avertissement est injecté
```

## F3 — Sessions et branches ([#22](https://github.com/skl1017/sensai/issues/22))

**US 3.1** — En tant qu'utilisateur, je veux reprendre une conversation après avoir fermé la CLI, afin de continuer là où je m'étais arrêté.

```gherkin
Scénario: Reprise
  Quand je redémarre la CLI et reprends la session "s1"
  Alors l'historique de la session est rechargé
```

**US 3.2** — En tant qu'utilisateur, je veux éditer un ancien message sans perdre l'original, afin d'explorer une autre piste.

```gherkin
Scénario: Fork
  Quand j'édite le message du tour 1
  Alors une branche sœur est créée et la branche originale reste navigable
```

**US 3.3** — En tant qu'utilisateur, je veux interrompre une réponse, afin de ne pas attendre une réponse inutile, sans corrompre mon historique.

```gherkin
Scénario: Ctrl-C
  Quand j'appuie sur Ctrl-C pendant une réponse
  Alors la génération s'arrête et la session reste intacte
```

## F4 — Personnalisation et contexte ([#27](https://github.com/skl1017/sensai/issues/27))

**US 4.1** — En tant qu'opérateur, je veux changer la persona de l'agent par un fichier YAML, afin d'adapter son rôle et son ton sans code.

```gherkin
Scénario: Changement de persona
  Quand state["persona"] passe de "default" à "tutor"
  Alors le prompt système change et l'historique est conservé
```

**US 4.2** — En tant qu'utilisateur, je veux que l'agent applique mes préférences à chaque tour, afin de ne pas les répéter.

```gherkin
Scénario: Profil
  Étant donné un profile.yaml contenant "réponds en français"
  Quand j'envoie un message
  Alors les instructions du profil sont dans le prompt système
```

**US 4.3** — En tant qu'utilisateur, je veux tenir de longues conversations sans que l'agent oublie l'essentiel, afin de ne pas devoir le lui rappeler.

```gherkin
Scénario: Compression
  Étant donné un historique au-delà de 80 % du budget de tokens
  Quand un nouveau tour arrive
  Alors les vieux tours sont résumés et les noms, chiffres et décisions sont conservés
```

## F5 — Contrôle humain ([#33](https://github.com/skl1017/sensai/issues/33))

**US 5.1** — En tant qu'utilisateur, je veux que l'agent me pose une question quand il lui manque une information, afin qu'il ne devine pas.

```gherkin
Scénario: Précision demandée
  Quand je demande "réserve-moi un train" sans date
  Alors l'agent me demande la date et utilise ma réponse
```

**US 5.2** — En tant qu'utilisateur, je veux valider toute action à effet de bord, afin que rien ne se passe sans mon accord.

```gherkin
Scénario: Refus
  Quand le modèle appelle write_file et que je réponds "no"
  Alors rien n'est écrit et le refus revient comme observation
```

**US 5.3** — En tant qu'opérateur, je veux qu'un agent lancé sans utilisateur refuse par défaut, afin qu'un oubli de configuration ne le rende pas dangereux.

```gherkin
Scénario: HeadlessUI
  Étant donné l'agent lancé avec HeadlessUI
  Quand un outil à approbation est appelé
  Alors l'action est refusée
```

## F6 — Outils système ([#38](https://github.com/skl1017/sensai/issues/38))

**US 6.1** — En tant qu'utilisateur, je veux que l'agent lise et écrive des fichiers de mon projet, afin de l'aider sur mon code ou mes notes.

```gherkin
Scénario: Écriture approuvée
  Quand le modèle appelle write_file et que j'accepte
  Alors le fichier est créé dans la racine autorisée
```

**US 6.2** — En tant qu'opérateur, je veux restreindre les accès disque et réseau de l'agent, afin qu'il ne puisse pas sortir de son périmètre.

```gherkin
Scénario: Traversée de chemin
  Quand read_file reçoit "../../etc/passwd"
  Alors l'accès est refusé avec la racine autorisée

Scénario: SSRF
  Quand fetch_page reçoit "http://localhost:11434"
  Alors la requête est refusée
```

**US 6.3** — En tant qu'utilisateur, je veux que l'agent cherche sur le web en citant ses URL, afin d'avoir des informations à jour et vérifiables.

```gherkin
Scénario: Recherche
  Quand le modèle appelle web_search
  Alors chaque résultat porte une URL citable
```

## F7 — MCP ([#43](https://github.com/skl1017/sensai/issues/43))

**US 7.1** — En tant qu'opérateur, je veux brancher des serveurs MCP existants, afin de donner de nouveaux outils à l'agent sans les coder.

```gherkin
Scénario: Serveur tiers
  Étant donné le serveur MCP "fs" configuré
  Quand l'agent démarre
  Alors ses outils sont disponibles sous le nom "fs__<outil>"

Scénario: Outil à effet de bord
  Quand le modèle appelle un outil MCP absent de read_only
  Alors une approbation est demandée
```

**US 7.2** — En tant que développeur, je veux exposer les outils de Sensai via MCP, afin de les réutiliser depuis d'autres clients.

```gherkin
Scénario: Serveur maison
  Étant donné le serveur MCP de Sensai lancé
  Quand un client externe appelle calculator("2+3")
  Alors il reçoit "5"
```

## F8 — RAG ([#53](https://github.com/skl1017/sensai/issues/53))

**US 8.1** — En tant qu'utilisateur, je veux poser des questions sur mes documents, afin d'obtenir des réponses fondées sur leur contenu.

```gherkin
Scénario: Question sur un document
  Étant donné un dossier de documents txt, md et pdf ingéré
  Quand je pose une question dont la réponse s'y trouve
  Alors la réponse s'appuie sur le passage concerné
```

**US 8.2** — En tant qu'utilisateur, je veux que l'agent cite ses sources, afin de vérifier ses réponses.

```gherkin
Scénario: Citation
  Quand l'agent répond à partir d'un document
  Alors la réponse cite l'identifiant du passage utilisé
```

**US 8.3** — En tant qu'opérateur, je veux que seuls les passages pertinents soient injectés, afin d'éviter les réponses polluées par du hors-sujet.

```gherkin
Scénario: Seuil
  Étant donné min_score = 0.5
  Quand un passage obtient 0.3 après reranking
  Alors il n'apparaît pas dans le contexte
```

## F9 — Multi-agents ([#77](https://github.com/skl1017/sensai/issues/77))

**US 9.1** — En tant que développeur, je veux qu'un agent délègue une tâche à un sous-agent spécialisé, afin de composer des agents avec leur propre persona et leurs propres outils.

```gherkin
Scénario: Délégation
  Quand le modèle appelle ask_researcher("résume ce sujet")
  Alors le sous-agent répond et sa réponse revient comme observation
```

**US 9.2** — En tant qu'opérateur, je veux borner les appels entre agents, afin d'éviter récursions infinies et dérives de coût.

```gherkin
Scénario: Profondeur maximale
  Étant donné une profondeur maximale de 2
  Quand un sous-agent de profondeur 2 appelle un autre agent
  Alors l'appel est refusé
```
