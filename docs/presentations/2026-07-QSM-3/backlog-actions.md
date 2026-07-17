<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# Backlog — tout ce qu’il reste à faire pour un BPS « full open source »

**Date :** 16 juillet 2026  
**Analyse détaillée :** [analyse-open-source-readiness.md](analyse-open-source-readiness.md)  
**Usage :** répartition du travail (workstreams + owners + priorités). Base pour les prochaines slides.

Légende priorité : **P0** bloquant open-source ready v1 · **P1** fort · **P2** important · **P3** nice-to-have / v2  
Légende effort : **S** &lt; 1 j · **M** quelques jours · **L** semaines · **XL** multi-semaines / ESA

Owners indicatifs (à confirmer) : **ACRI** ACRI-ST · **ARE** Aresys · **ESA** · **SME** processor leads · **COM** community / externals

---

## Vue d’ensemble — workstreams

```text
WS1 Fondation repo     → merge #24 #30 #36, licence GitHub, SECURITY, CITATION
WS2 Documentation      → rebase/merge #35, ATBD/SUM web, PDF export, triage docs
WS3 CI & usine         → CI verte, branch protection, décision GitLab, tests réels
WS4 Distribution       → bundle public, AUX policy, L1F packaging, PyPI/Zenodo (plus tard)
WS5 Licence & supply   → SRF, NOTICE, deps BIPR/EOCFI/MKL, SBOM
WS6 Communauté         → triage issues, Discussions, good-first-issue, office hours
WS7 Science produit    → bugs/qualité issus du terrain (#11, #13, etc.)
```

Les workstreams **1–3** sont surtout ACRI ; **4–5** ESA+Aresys ; **6** ACRI+ESA ; **7** SME+Aresys.

---

## WS1 — Fondation dépôt (P0)

| ID | Action | Issue/PR | Prio | Effort | Owner | Statut |
|---|---|---|---|---|---|---|
| F1 | Merger **PR #24** — CODEOWNERS, Dependabot, `REUSE.toml`, `LICENSES/` | #20 / #24 | P0 | S | ACRI | Ouvert mergeable |
| F2 | Débloquer **PR #30** (REUSE, pre-commit, Dependabot governance) puis merger | #21 / #30 | P0 | M | ACRI | CI rouge |
| F3 | Finaliser et merger **PR #36** (SPDX inline) après #24 | #25 / #36 | P1 | M | ACRI | Draft |
| F4 | Remplacer / ajouter `LICENSE` détectable Apache-2.0 (finir `NOASSERTION` GitHub) | — | P0 | S | ACRI | À faire |
| F5 | Ajouter `SECURITY.md` (lien advisory + email) | docs About | P0 | S | ACRI | Annoncé, absent |
| F6 | Ajouter `CITATION.cff` | docs About | P1 | S | ACRI+ESA | Annoncé, absent |
| F7 | Ajouter topics GitHub (`biomass`, `sar`, `esa`, `open-science`, …) | — | P2 | S | ACRI | Vide |
| F8 | Vérifier règlesets branch protection `develop` / `main` alignés avec la doc | doc CI | P0 | M | ACRI+ESA | Doc ≠ repo |
| F9 | Clôturer typos / micro-docs : choisir **une** PR (#49 ou #50), fermer le doublon ; merger #44 via #35 | #47 #44 | P2 | S | ACRI | Doublons ouverts |

**Ordre recommandé :** F1 → F2 → F3 → puis F4–F6 en parallèle des reviews.

---

## WS2 — Documentation & site (P0 / P1)

| ID | Action | Issue/PR | Prio | Effort | Owner | Statut |
|---|---|---|---|---|---|---|
| D1 | Rebase **PR #35**, résoudre conflits, merger dans `develop` | #22 / #35 | P0 | M | ACRI | CONFLICTING |
| D2 | S’assurer que le déploiement biomass-disc.info suit `develop` (plus seulement la branche docs) | — | P0 | S–M | ACRI | À vérifier |
| D3 | Corriger export PDF ATBD AGB (`\tag` dans `split`) | #39 | P1 | M | ACRI | Ouvert |
| D4 | Convertir ATBD **FH** en web (même modèle que AGB) | (créer issue) | P1 | L | ACRI+SME | Annoncé, pas d’issue |
| D5 | Convertir ATBD **FD** en web | (créer issue) | P1 | L | ACRI+SME | Annoncé, pas d’issue |
| D6 | Convertir / stub web **SUM** (au moins navigation + chapitres critiques) | (créer issue) | P1 | XL | ACRI+ARE | PDF only |
| D7 | Convertir ATBD L1 SAR + Stack (ou prioriser selon ESA) | (créer issues) | P2 | XL | ACRI+SME | PDF only |
| D8 | Décider du sort de #41 (MyST → RST) : reporter ou refuser (coût vs bénéfice) | #41 | P3 | L | ACRI | Ouvert non priorisé |
| D9 | Décider du modèle APT (#34) : spike + ADR, ou park | #34 | P2 | M | ACRI+ESA | Ouvert |
| D10 | Aligner chemins PDF portail (`BPS_v4.4.2` → `BPS_v4_4_4`) | applicable-documents | P2 | M | ARE/ESA | Noté dans docs |
| D11 | Page « Known issues » multi-versions (ex. NumPy / 4.3.1) | #11 | P2 | S | ACRI+ARE | Demandé |

---

## WS3 — CI / CD & migration GitLab (P0)

| ID | Action | Issue/PR | Prio | Effort | Owner | Statut |
|---|---|---|---|---|---|---|
| C1 | Faire passer baseline PR #30 au vert (REUSE + pre-commit + dependabot gate) | #30 | P0 | M | ACRI | Rouge |
| C2 | Remplacer placeholders `test/baseline|extended|heavy` par **vrais** tests unitaires d’au moins 1–2 processors | — | P1 | L | ACRI+ARE | Scaffold only |
| C3 | Activer / lander `release.yml` (build, SBOM, GitHub Release) | doc release | P1 | M | ACRI | Doc only |
| C4 | **Décision écrite** dual-track GitLab vs GitHub : sunset date, ou matrice « quoi tourne où » publiée dans docs | — | P0 | M | ACRI+ARE+ESA | Non documenté |
| C5 | Porter progressivement jobs packaging/testplan utiles vers GitHub Actions **ou** exposer badges/status GitLab aux contributeurs | `.gitlab-ci.yml` | P1 | XL | ARE+ACRI | GitLab dominant |
| C6 | Activer DCO check + required status checks sur `develop` | — | P0 | S | ACRI | Dépend F2/F8 |
| C7 | Documenter clairement que les tests « Heavy » restent maintainer-only / infra limitée | doc CI | P2 | S | ACRI | Partiel |

---

## WS4 — Distribution & reproductibilité (P0 ESA)

| ID | Action | Issue/PR | Prio | Effort | Owner | Statut |
|---|---|---|---|---|---|---|
| R1 | Politique publique download **AUX** (GMF, IRI, FNF, LCM, CAL_AB) : libre, gated, ou MAAP-only + critères | #2 | P0 | L | ESA+ARE | Discussion ouverte |
| R2 | Documenter et lier depuis README/tutorial la procédure exacte (MAAP paths + éventuel portail) | #2 | P0 | M | ARE+ACRI | Commentaires partiels |
| R3 | Clarifier accès **BPS delivery bundle** (plus seulement `service.aresys.it` dans un fil d’issue) | #2 comments | P0 | M | ARE+ESA | Ad hoc |
| R4 | Publier packaging **L1 Framing** (conda/tarball) ou retirer L1F du tutorial utilisateur jusqu’à dispo | #40 | P0 | L | ARE | Confirmé manquant |
| R5 | Première **GitHub Release** officielle alignée version suite | — | P1 | M | ACRI+ESA | Aucune release |
| R6 | Activer publication **PyPI** (OIDC) quand packages prêts | doc release | P2 | L | ACRI | Commenté |
| R7 | **Zenodo** DOI automatique sur release | slides / roadmap | P2 | M | ACRI+ESA | Roadmap |
| R8 | Documenter rebuild from-source vs usage binaires (`bps-*-binaries`) | — | P1 | M | ARE+ACRI | Opaque |

---

## WS5 — Licence, SBOM, supply chain (P0 / P1)

| ID | Action | Issue/PR | Prio | Effort | Owner | Statut |
|---|---|---|---|---|---|---|
| L1 | Publier **SRF** (Software Reuse File) et remplacer le lien vide dans `CREDITS.md` | CREDITS | P0 | M | ARE+ESA | Lien `()` |
| L2 | `NOTICE` / inventaire SPDX des `LicenseRef-*` (ARESYS-BIPR, EOCFI, ISSL, IRI) | — | P0 | M | ARE+ACRI | Partiel CREDITS |
| L3 | Clarifier ce qui est redistribuable sous Apache-2.0 vs composants non redistribuables | licensing docs | P0 | M | ESA+ARE | Critique « full OSS » |
| L4 | SBOM CycloneDX attaché à chaque release GitHub | doc release | P1 | M | ACRI | Dépend C3 |
| L5 | Gate licence deps dans CI (déjà prévu Dependabot + review) | #20 | P1 | S | ACRI | PR #24 |
| L6 | Harmoniser métadonnées conda/`pyproject` (certains `meta.yaml` disent encore MIT) | recipes | P2 | M | ARE | Incohérent |

---

## WS6 — Communauté & hygiene backlog (P1)

| ID | Action | Issue/PR | Prio | Effort | Owner | Statut |
|---|---|---|---|---|---|---|
| H1 | Triage batch : labels `type:*`, `needs-triage`/`needs-discussion`, composants sur #2 #11 #12 #13 #40 | issues ouvertes | P0 | S | ACRI+ARE | Non labelisées |
| H2 | Déplacer #12 (et Q&A similaires) vers **Discussions** | #12 | P1 | S | ACRI | Issue mal typée |
| H3 | Répondre ou assigner SME sur #13 (FH noisy) | #13 | P1 | M | SME | 0 commentaire |
| H4 | Créer 5+ **good-first-issue** non triviaux (docs ATBD typo, tests, CI docs) | — | P1 | M | ACRI | Quasi vide |
| H5 | Annoncer **Office Hours** ou prochain community meeting (date) | channels.md | P1 | S | ACRI+ESA | « not scheduled » |
| H6 | Remplir Discussions (pinned FAQ : bundle, AUX, MAAP, versions) | Discussions | P1 | M | ACRI+ARE | 2 fils only |
| H7 | Project board public « BIOMASS BPS Backlog » (issue #23 fermée — vérifier si board existe et est utiliséé) | #23 | P2 | S | ACRI | Issue fermée |
| H8 | Guidelines « AI-assisted PR » (ex. #50) si besoin de politique | — | P3 | S | ACRI | Optionnel |

---

## WS7 — Produit / science (hors fondation, à planifier)

| ID | Action | Issue/PR | Prio | Effort | Owner | Statut |
|---|---|---|---|---|---|---|
| S1 | Documenter workaround NumPy pour 4.3.1 + statut fix dans versions récentes | #11 | P2 | S | ARE | Ouvert |
| S2 | Clarifier comportement attendu FH L2A (speckle vs bug vs AUX) | #13 | P1 | M | SME | Ouvert |
| S3 | Suivre campagne L1A→L1C (réponse catalogue) | #12 | P2 | S | ESA | Ouvert |
| S4 | Continuer sync JIRA/GitLab interne → GitHub pour bugs visibles communauté | BPS-* PRs | P2 | L | ARE | Opaque |

---

## Checklist « Open Source Ready v1 » (DoD)

Cocher quand c’est **sur `develop` + documenté + annoncé**.

### Must (P0)

- [ ] PR #24 mergée
- [ ] PR #30 verte et mergée
- [ ] PR #35 rebasée et mergée ; site docs suit `develop`
- [ ] Licence GitHub = Apache-2.0
- [ ] `SECURITY.md` présent
- [ ] Branch protection + checks requis sur `develop`
- [ ] Décision GitLab/GitHub publiée
- [ ] Policy AUX + bundle documentée (même si gated) ; #2 close ou convertie
- [ ] Position claire sur L1F packaging (#40)
- [ ] SRF / NOTICE publics
- [ ] Issues communauté labelisées

### Should (P1)

- [ ] Première GitHub Release + SBOM
- [ ] `CITATION.cff`
- [ ] PDF ATBD AGB CI vert (#39)
- [ ] Au moins un ATBD web supplémentaire (FH ou FD) démarré
- [ ] 5 good-first-issue
- [ ] Office Hours ou meeting annoncé
- [ ] Tests unitaires réels dans baseline (plus que placeholders)

### Later (P2+)

- [ ] PyPI
- [ ] Zenodo DOI
- [ ] SUM web
- [ ] Tous ATBD web
- [ ] Suppression / remplacement composants non permissifs (v2 licence)

---

## Matrice de répartition rapide (pour slides)

| Qui | Prend |
|---|---|
| **ACRI-ST** | F1–F9, D1–D3, D8–D9, C1–C3, C6–C7, H1–H8, partie L4–L5 |
| **Aresys** | R2–R4, R8, L1–L3, L6, C4–C5, S1–S2, sync JIRA |
| **ESA** | R1, R3, R5 gate, L3, C4, H5, S3, validation DoD v1 |
| **SME / processor leads** | D4–D7 reviews, S2, CODEOWNERS reviews post-#24 |
| **Communauté externe** | good-first-issue, typos, Discussions Q&A, tests docs |

---

## Quick wins (semaine 1)

1. Merger #24 (F1).  
2. Labels sur toutes les issues ouvertes (H1).  
3. Choisir une PR typo #47 et merger (F9).  
4. Ajouter `SECURITY.md` + fixer fichier licence (F4, F5).  
5. Réponse courte publique sur #2 / #40 (« status + owner + next date ») même sans solution finale (R1/R4).  
6. Plan de rebase #35 (D1) écrit dans la PR.

## Jalons suggérés

| Jalon | Contenu |
|---|---|
| **M1 — Foundations landed** | #24 + #30 sur `develop`, protection branche, licence OK |
| **M2 — Docs on default branch** | #35 mergée, site = `develop` |
| **M3 — Reproducible path documented** | AUX/bundle/L1F policy publiée |
| **M4 — First OSS release** | GitHub Release + SBOM + CITATION |
| **M5 — Community loop** | Office hours + 5 GFI + Discussions FAQ |

---

## Hors scope explicite (ne pas promettre dans les slides v1)

- Rebuild 100 % from source sans binaires Aresys / sans MKL / sans EOCFI.  
- Relicencer SARFOC/SARINT en Apache.  
- Fermeture immédiate de GitLab.  
- Migration forcée Markdown → RST (#41).  
- SUM et tous les ATBD web complets.

Ces items appartiennent à une **v2 « deep open »** après DoD v1.
