<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# Analyse — readiness open source BIOMASS BPS

**Date :** 16 juillet 2026  
**Repo :** [github.com/BioPAL/BPS](https://github.com/BioPAL/BPS)  
**Branche analysée :** écart entre `develop` (défaut), PRs ouvertes, branche `docs/sphinx-site-migration`, et la doc publiée sur [biomass-disc.info/docs](https://www.biomass-disc.info/docs/)  
**Objectif :** mesurer l’écart entre le discours « projet open source opérationnel » et ce qui est réellement livré, pour préparer une présentation de répartition du travail.

Document compagnon : [backlog-actions.md](backlog-actions.md).

---

## 1. Verdict en une phrase

BPS est **ouvert en lecture** (repo public, licence Apache 2.0 affichée) et dispose d’un **programme open source très abouti sur le papier**, mais n’est **pas encore « full open source » opérationnel** : la majorité des fondations (REUSE, CI GitHub à tiers, site Sphinx, releases, données AUX, packaging public) est **écrite ou ouverte en PR**, pas **mergée et utilisable** sur `develop`.

---

## 2. Méthode

| Source | Ce qui a été couvert | Limite |
|---|---|---|
| **GitHub Issues** | 14 ouvertes + 12 fermées (toutes listées) | — |
| **GitHub PRs** | 6 ouvertes + historique merged/closed | — |
| **Discussions** | 2 fils (welcome + how to ask) | Très peu d’usage communauté |
| **Docs site** (`docs/` sur branche Sphinx + meeting 30 juin) | Contributing, governance, licensing, science/user guide, channels, release process, CI tiers | Beaucoup décrit un état cible, pas `develop` |
| **Arborescence Git** | `develop` vs `main` vs branches fondation / CI / docs | — |
| **GitLab Issues** | Non accessibles depuis cet environnement (`glab` absent, pas d’URL publique BioPAL/BPS sur GitLab) | Voir §3 |
| **CREDITS / binaires / CI legacy** | `CREDITS.md`, `bps-*-binaries`, `.gitlab-ci.yml` | SRF lié vide dans CREDITS |

---

## 3. Note GitLab vs GitHub

Le suivi **public** des contributions est sur **GitHub**. Il reste pourtant une **CI GitLab complète** (`.gitlab-ci.yml`, packaging conda, dashboards) et des tickets internes visibles dans les titres de PR (`BPS-959`, etc. — style JIRA Aresys/ESA).

**Interprétation pour la présentation :**

- GitHub = face open source / communauté.
- GitLab (+ JIRA) = usine de build / suivi contractuel encore actifs.
- « Full open source » implique de **clarifier la migration** : soit GitHub devient la seule usine, soit on documente explicitement le dual-track et ce que les contributeurs externes ne voient pas.

Sans accès au tracker GitLab interne, le backlog public ci-dessous est **GitHub-centré**. Les tickets GitLab/JIRA non exposés restent un risque de double backlog.

---

## 4. État réel du dépôt (juillet 2026)

### 4.1 Ce qui est vrai sur `develop` (branche par défaut)

| Élément | Statut |
|---|---|
| Code processors (`bps-*`) | Présent, public |
| Templates Issues / PR | Présents (via PR #19 mergée) |
| Dossier `docs/` Sphinx | **Absent** de `develop` |
| `LICENSES/`, `REUSE.toml`, CODEOWNERS, Dependabot | **Absents** (dans PR #24) |
| Workflows CI (tiers, DCO, REUSE, tests) | **Absents** (dans PR #30) |
| Workflow docs | Seulement sur branche docs |
| GitHub Releases / tags publics de release OSS | Aucune release GitHub listée |
| Licence détectée par GitHub | `NOASSERTION` (LICENSE.md non reconnu comme SPDX standard) |
| Topics / CITATION.cff / SECURITY.md à la racine | Absents / annoncés « to be added » |

### 4.2 `main` encore plus minimal

Pas de dossier `.github/` sur `main` (404 API). La promotion `develop → main` OSS (gate ESA décrite dans la doc) n’est pas encore matérialisée par les artefacts open-source.

### 4.3 PRs fondation — le vrai chantier

| PR | Issue | Contenu | Taille | État |
|---|---|---|---|---|
| **#24** | #20 | CODEOWNERS, Dependabot, `REUSE.toml`, `LICENSES/*` | +433 / 5 fichiers | Ouverte, mergeable, **pas mergée** |
| **#30** | #21 | CI complète (tier policy, baseline/extended/heavy, pre-commit, tests placeholder) | +1894 / 17 fichiers | Ouverte ; **CI gate rouge** (REUSE, pre-commit, Dependabot governance) |
| **#35** | #22 | Migration site Sphinx + ATBD AGB + tutorial + presentations | +27840 / 206 fichiers | Ouverte ; **CONFLICTING** avec `develop` |
| **#36** | #25 | Headers SPDX inline (~800 fichiers) | +2472 / 821 fichiers | **Draft** |
| **#49 / #50** | #47 | Typo README `runnin` → `running` | 1 ligne | Doublon (interne + externe) |
| ~~#45~~ | #44 | Meeting « held » | Mergé **dans** `docs/sphinx-site-migration`, pas dans `develop` | |

**Lecture :** la doc et le First Developer Meeting décrivent déjà CODEOWNERS, REUSE bloquant, tiers CI, site unique — alors que sur `develop` seuls les **templates** sont arrivés. L’écart narration / réalité est le message central des prochaines slides.

### 4.4 Écart doc ↔ dépôt

La documentation (branche docs / site live) affirme notamment :

- REUSE = gate bloquante CI ;
- 10 checks baseline parallèles, tiers 0/1/2 ;
- release.yml + SBOM + GitHub Release ;
- PyPI « préparé mais commenté » ;
- Zenodo DOI « on roadmap » ;
- Office Hours / mailing list « coming soon » ;
- `CITATION.cff` et `SECURITY.md` « will be / to be added ».

Or ces mécanismes ne sont **pas sur la branche par défaut**. Un contributeur qui clone `develop` aujourd’hui ne retrouve pas le projet décrit dans les slides du 30 juin.

---

## 5. Cartographie des issues GitHub

### 5.1 Fondation open source (équipe programme / ACRI-ST)

| # | Titre | Labels | Rôle |
|---|---|---|---|
| **20** | Repository config foundation (CODEOWNERS, Dependabot, REUSE, LICENSES) | approved | Prérequis licence / review |
| **21** | Migrate complete CI/CD pipeline | approved | Prérequis confiance merge |
| **22** | Migrate documentation site into the repository | approved | Prérequis « one repo, one site » |
| **25** | Migrate REUSE bulk → inline SPDX | approved | Qualité licence long terme |
| **34** | APT-inspired ATBD authoring model | needs-triage | Vision science web |
| **39** | Bug export PDF ATBD (`\tag` dans `split`) | needs-triage | Qualité artefact science |
| **41** | Migrer MyST/Markdown → reStructuredText | needs-triage | Refacto docs (discutable / coût élevé) |
| **44** | Marquer First Dev Meeting comme held | approved, good-first-issue | Cosmétique (déjà fait sur branche docs) |
| **47** | Typo README | approved | Micro-fix (PRs #49/#50) |

Issues fermées #23, #26–#29 = découpage initial du CI (absorbé dans #21 / PR #30).

### 5.2 Communauté / usage (souvent sans labels, peu triées)

| # | Sujet | Signal open source |
|---|---|---|
| **2** | Liens download GMF, IRI, FNF, LCM, CAL_AB | **Bloquant reproductibilité** — discussion ESA/MAAP en cours, pas de distribution libre hors process |
| **11** | NumPy > 2.0 sur BPS 4.3.1 | Docs « known issues » demandée ; versions legacy |
| **12** | Zones L1A vs L1C (campagne reprocessing) | Question catalogue / mission, pas code |
| **13** | FH L2A « noisy » | Question scientifique sans réponse (0 commentaire) |
| **40** | Accès `bps_l1_framing_processor` via conda/bundle | Framing absent du packaging utilisateur ; Aresys confirme exploration |

Issues fermées #1, #3–#6, #9–#10 = bugs runtime / install / SUM — preuve qu’il y a déjà des utilisateurs, mais le support reste ad hoc.

### 5.3 Hygiene de triage

Problèmes récurrents :

- Issues communauté **sans template labels** (`needs-triage`, `type:*`, composant).
- Plusieurs issues devraient être des **Discussions** (#12, parties de #13) — les Discussions existent mais sont quasi vides.
- `good-first-issue` : peu d’items réellement prêts hors typos.
- Doublons de PRs sur le même typo (#49 vs #50) : signe que le flux externe démarre, mais sans merge rapide ni CI sur `develop`.

---

## 6. Lecture de la documentation web (ce qu’elle promet)

### 6.1 Points forts déjà rédigés (capital à conserver)

- Narratif Open Science / FAIR / Apache 2.0 / pas d’assignment de copyright.
- Parcours contributeur en 5 étapes avec **gate d’approbation**.
- Gouvernance claire (ESA, maintainers, processor leads, CODEOWNERS cible).
- Guides licensing + dépendances + REUSE.
- Tutorial « Run BPS locally » + catalogue applicable documents.
- Premier ATBD web (L2 AGB) comme preuve de concept.
- First Developer Meeting (30 juin 2026) : onboarding communauté.

### 6.2 Promesses non tenues ou partielles

| Promesse docs / slides | Réalité |
|---|---|
| Site Sphinx = source of truth dans le repo | Branche + site live ; **pas sur `develop`** ; PR #35 en conflit |
| CI à tiers sur chaque PR | PR #30 rouge ; absent de `develop` |
| REUSE bloquant | Pas sur `develop` ; PR #36 draft |
| SUM web | Toujours PDF only |
| ATBD web FH / FD | Annoncés « planned », pas commencés en issues dédiées visibles |
| Office Hours / community meeting / mailing list | Explicitement « not scheduled yet » |
| Zenodo DOI | Roadmap |
| PyPI | Commenté dans release process |
| SECURITY.md / CITATION.cff | Annoncés, absents |
| SBOM à chaque release | Pipeline release pas actif côté GitHub Releases |
| Bundle + AUX « with delivery » (SUM) | Bundle Aresys ; AUX partiel MAAP ; issue #2 ouverte |

### 6.3 Science & user guides

- **User Guide** = pointeur SUM PDF (v4.4.1) — pas de SUM navigable.
- **Science Guide** = 1 ATBD web (AGB draft) + liste PDF pour L1/FH/FD/ICD/IODD/AUX.
- **Applicable documents** : table à jour v4.4.4, mais chemins PDF encore sous `BPS_v4.4.2/` (dette portail notée dans la page).

---

## 7. Freins « full open source » hors process GitHub

Même après merge des PRs #24/#30/#35, le projet ne sera pas entièrement redistribuable / rebuildable comme un OSS classique.

### 7.1 Licences et composants non permissifs (`CREDITS.md`)

| Composant | Licence | Impact |
|---|---|---|
| SARFOC / SARINT / parties L1 GPP | `LicenseRef-ARESYS-BIPR` | Propriétaire Aresys — pas redistribuable librement |
| codesynthesis | Commercial proprietary | Binding XML C++ |
| eocfi | EO CFI Terms (ESA) | Conditions spécifiques, pas Apache |
| intel_mkl/ipp | ISSL | Redistribution encadrée |
| IRI-2020 | Licence modèle dédiée | À tracer dans REUSE/NOTICE |

Le lien vers le **Software Reuse File (SRF)** dans CREDITS est **vide** `()`. Sans SRF public à jour, l’audit licence ESA/externe est incomplet.

### 7.2 Binaires et packaging

- Dossiers `bps-l1_binaries` / `bps-stack_binaries` : distribution native, pas un rebuild from-source trivial pour un contributeur externe.
- Packaging utilisateur toujours centré **bundle Aresys** (`service.aresys.it`) + conda interne / MAAP.
- L1 Framing volontairement hors conda (issue #40) — trou dans la chaîne L1F→L1 documentée dans le tutorial.

### 7.3 Données auxiliaires

Issue #2 + commentaires Aresys : AUX « internal resources » en discussion avec ESA ; disponibles surtout via **MAAP** ; pas de politique claire de download public. Sans AUX, le code open source ne reproduit pas les produits.

### 7.4 Dual CI

`.gitlab-ci.yml` reste l’usine conda/testplan. Tant que GitHub Actions n’a que des placeholders de tests, la « confiance OSS » est cosmétique : les vrais gates scientifiques restent hors vue des contributeurs GitHub.

---

## 8. Maturité par pilier (grille First Dev Meeting)

Échelle : **0** absent · **1** documenté · **2** en PR / partiel · **3** sur `develop` et utilisé · **4** mature (releases, communauté, métriques)

| Pilier | Score | Commentaire |
|---|---|---|
| **01 Document** | **2** | Excellent corpus sur branche docs / site ; pas mergé ; SUM/ATBD encore PDF-first |
| **02 Adapt** | **2** | Templates OK ; CODEOWNERS/REUSE/LICENSES en PR #24 |
| **03 Automate** | **1–2** | Design CI abouti ; PR #30 rouge ; GitLab encore dominant |
| **04 Empower** | **2** | Guides + templates ; peu de `good-first-issue` réels ; triage communauté faible |
| **05 Gather** | **1** | 1 meeting tenu ; Discussions vides ; office hours absents |

**Score global subjectif : ~45 %** du chemin « full open source distribuable » (process + licence + rebuild + données + communauté). Closer à **60–70 %** si on ne compte que « code lisible + intention ESA », et **~25 %** si on exige « clone, build, run, contribute sans accès Aresys/MAAP ».

---

## 9. Ce que la prochaine présentation doit raconter

Structure recommandée (slides à faire ensuite) :

1. **Écart narration / réalité** — une slide choc : ce que dit le site vs ce qu’il y a sur `develop`.
2. **Trois couches de dette** — (A) fondations repo à merger, (B) distribution/rebuild/AUX, (C) communauté & hygiene.
3. **Workstreams parallélisables** avec owners (voir backlog).
4. **Quick wins** (typos, labels, SECURITY.md, license file, merge #24) vs **ESA gates** (AUX, BIPR, L1F packaging, dual CI).
5. **Definition of Done « full open source v1 »** — checklist courte et partagée.

---

## 10. Definition of Done proposée — « Open Source Ready v1 »

Un jalon réaliste (pas le rêve « 100 % rebuild from source sans binaire »).

- [ ] PRs #24, #30 (verte), #35 mergées dans `develop` (ordre : config → CI → docs, ou rebase docs après config).
- [ ] `develop` protégé avec checks CI réels (même baseline minimale).
- [ ] Licence GitHub détectée Apache-2.0 (`LICENSE` / `LICENSE.txt` SPDX).
- [ ] `SECURITY.md`, `CITATION.cff`, NOTICE/SRF publics à jour.
- [ ] Au moins une **GitHub Release** avec SBOM + artefacts documentés.
- [ ] Page publique « How to obtain bundle + AUX » (même si gated) + issue #2 résolue ou convertie en policy.
- [ ] Backlog communauté labelisé ; questions non actionables déplacées en Discussions.
- [ ] Décision écrite sur le destin de GitLab CI (sunset date ou dual-track documenté).
- [ ] Au moins 5 `good-first-issue` non triviaux prêts.
- [ ] Office Hours ou créneau community meeting annoncé.

**Hors scope v1 (v2 / ESA) :** suppression LicenseRef-ARESYS-BIPR, rebuild full native sans MKL/EOCFI, PyPI + Zenodo, SUM 100 % web, tous les ATBD web.

---

## 11. Sources

- Issues / PRs : [github.com/BioPAL/BPS/issues](https://github.com/BioPAL/BPS/issues), [pulls](https://github.com/BioPAL/BPS/pulls)
- Docs : `docs/` (branche `docs/sphinx-site-migration`), [speaker-notes First Dev Meeting](../2026-06-30-first-dev-meeting/speaker-notes.md)
- Crédits / licences : `CREDITS.md`, `LICENSE.md`, pages `docs/about/licensing/`
- CI legacy : `.gitlab-ci.yml`
