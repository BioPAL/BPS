# BIOMASS BPS Governance




## The Project

The BIOMASS Processing Suite (BPS) Project (The Project) is an industrial software project affiliated with the European Space Agency (ESA) and BIOMASS satellite mission. The goal of The Project is to develop, maintain and evolve the software suite for processing Level 1 (L1), Level 2A and Level 2B data of ESA's BIOMASS mission. The Project is developed by Aresys and ACRI-ST under ESA contract, and follows Open Science principles to support broader scientific community engagement.

The Software developed by The Project is released under the **Apache License 2.0**, a permissive open-source license that ensures transparency, legal clarity, and broad usability.

For complete information about licensing requirements, license compatibility, external dependencies, and legal obligations for contributors, please see the [Licensing documentation](../about/licensing.md).

The Project is developed by a team of distributed developers, called Contributors. Contributors are individuals who have contributed code, documentation, designs or other work to one or more Project repositories. Anyone can be a Contributor. Contributors can be affiliated with any legal entity or none.

The Project Community consists of all Contributors and Users of the Project. Contributors work on behalf of and are responsible to the larger Project Community and we strive to keep the barrier between Contributors and Users as low as possible.

The Project is formally affiliated with the European Space Agency (ESA) ([https://www.esa.int/](https://www.esa.int/)).

---

## Governance Structure

Our community is structured as a virtual organization. Authority is primarily distributed to both volunteer and employed community members irrespective of employment affiliation as they show their ability through contributions to The Project. The Project also seeks to debias this system of distributing authority through active interventions that engage and encourage participation from diverse communities.

The foundations of Project governance are:
- **Openness & Transparency**
- **Active Contribution**
- **Institutional Neutrality**

---

## Benevolent Dictator for Life (BDFL)

The ultimate decision-maker is the ESA BIOMASS Data Quality Manager: Clement Albinet, who has the final say in the case of disputes. Additionally, up to three "Benevolent Dictators for Life" (BDFL) can be appointed by ESA and in accordance to the BIOMASS Mission Manager. The BDFL model is followed by many successful open source projects. BIOMASS BPS will have at least one member of the ESA BIOMASS Mission Staff in this role: either the BIOMASS Data Quality Manager, Mission Manager, PDGS Manager or Science Manager.

As Dictator, the BDFL has the authority to make all final decisions for The Project under the BIOMASS mission manager. As Benevolent, the BDFL, in practice chooses to defer that authority to the consensus of the community discussion channels and the Steering Council.

**Current BDFL:**
- Clement Albinet (ESA)
---

## The Steering Council

The Project will have a Steering Council that consists of Project Contributors nominated as explained below. The Steering Council should be composed of a diverse array of backgrounds, viewpoints and talents. The overall role of the Council is to ensure, through taking input from the Community, the long-term well-being of The Project, scientifically, technically and as a community.

During the everyday Project activities, Council Members participate in all discussions, code review and other Project activities as peers with all other Contributors and the Community. In these everyday activities, Council Members do not have any special power or privilege through their membership on the Council.

The Steering Council and its Members play a special role in certain situations. In particular, the Council may, if necessary:
- Make decisions about the overall scope, vision and direction of The Project.
- Make decisions about strategic collaborations with other organizations or individuals.
- Make decisions about specific technical issues, features, bugs and pull requests. They are the primary mechanism of guiding the code review process and merging pull requests.
- Make decisions about the Services that are run by The Project and manage those Services for the benefit of The Project and Community.
- Make decisions when regular Community discussion does not produce consensus on an issue in a reasonable time frame.

**Current Steering Council Members:**

**Project Development and Maintenance** (alphabetical by first name):
- Giovanni Amoroso (Aresys)
- Klaus Scipal (ESA)
- Michele Caccia (ESA)
- Riccardo Piantanida (Aresys)
- Yoann Rey-Ricord (ACRI-ST)

**Scientific Advisory Board** (alphabetical by first name):
- Maciej Soja (WENR)

**Retired steering council members:**
- Davide Giudici (Aresys)
- Simone Mancon (Aresys)

---

## Institutional Partners

ESA is the ultimate decision-maker on the Project. The Steering Council is appointed on behalf of ESA to act as the primary leadership for The Project in day to day activities. No outside institution, individual or legal entity has the ability to own, control, usurp or influence The Project other than by participating in The Project as Contributors and Council Members.

An Institutional Partner is any recognized legal entity anywhere in the world that employs at least one Institutional Contributor or Institutional Council Member. Institutional Partners can be for-profit or non-profit entities.

**Current Institutional Partners (alphabetical):**
- [ACRI-ST](https://www.acri-st.fr/en/)
- [AresysATP](https://www.aresys.it/) 
- [CESBIO](https://www.cesbio.cnrs.fr/)
- [University of Chalmers](https://www.chalmers.se/)
- [CNRS](https://crbe.cnrs.fr/)
- [DLR](https://www.dlr.de/EN/Home/home_node.html) 
- [DTU](https://www.dtu.dk/)
- [ESA](https://www.esa.int/) 
- [GFZ](https://www.gfz.de/)
- [Mj Soja Consulting](http://mjsoja.com) 
- [Politecnico Milano](https://www.polimi.it/en/) 
- [S[&]T](https://www.stcorp.nl/) 
- [Wageningen University & Research](https://www.wur.nl/)

---

## Governance Overview

### Purpose

The governance framework ensures that BIOMASS BPS evolves in a controlled, transparent, and scientifically robust way, while remaining aligned with ESA rules, industrial constraints, and Open Science principles.

### Key Principles

- **Transparency**: Decisions are documented and reproducible
- **Scientific Rigor**: Changes are validated and traceable
- **Community Engagement**: External contributions are welcome and supported
- **Operational Stability**: Production code remains reliable and performant
- **FAIR Compliance**: Follows Findable, Accessible, Interoperable, Reusable principles

### Governance Structure

The following diagram shows the high-level governance structure:

```{mermaid}
graph TD
    A[ESA<br/>Mission Owner] --> B[Open Science Lead<br/>Strategy & FAIR]
    B --> H[Community<br/>Coordination]
    A --> C[Core Maintainers<br/>Technical Review]
    B --> C
    D[Scientific Experts<br/>Biomass/Structure/Uncertainty] --> C
    E[Contributors<br/>PR Submission] --> C
    C --> F[Repository<br/>Main Branch]
    A --> G[Final Authority<br/>Tier 2-3]
    
    classDef defaultStyle fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px,color:#333
    class A,B,C,D,E,F,G,H defaultStyle
```

The following diagram provides a more detailed view of the hierarchy and relationships between roles:

```{mermaid}
graph TB
    subgraph FinalAuthority["Final Authority"]
        ESA[ESA<br/>Mission Owner<br/>Final Authority]
    end
    
    subgraph Governance["Governance"]
        BDFL[BDFL<br/>Clement Albinet]
        SC[Steering Council<br/>Development & Scientific]
    end
    
    subgraph Operational["Operational"]
        OSL[Open Science Lead<br/>Strategy & FAIR]
        CM[Core Maintainers<br/>Technical Review]
        SME[Scientific Module Experts<br/>Scientific Validation]
    end
    
    subgraph Community["Community"]
        CONT[Contributors<br/>PR Submission]
        USERS[Users<br/>End Users]
    end
    
    ESA --> BDFL
    ESA --> SC
    BDFL --> SC
    SC --> OSL
    SC --> CM
    OSL --> CM
    SME --> CM
    CONT --> CM
    CM --> SC
    USERS --> CONT
    
    classDef defaultStyle fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px,color:#333
    class ESA,BDFL,SC,OSL,CM,SME,CONT,USERS defaultStyle
    
    style FinalAuthority fill:#ffffff,stroke:#9e9e9e,stroke-width:2px,color:#333
    style Governance fill:#ffffff,stroke:#9e9e9e,stroke-width:2px,color:#333
    style Operational fill:#ffffff,stroke:#9e9e9e,stroke-width:2px,color:#333
    style Community fill:#ffffff,stroke:#9e9e9e,stroke-width:2px,color:#333
    
    style ESA fill:#e1bee7,stroke:#e1bee7,stroke-width:3px
    style BDFL fill:#e1bee7,stroke:#e1bee7,stroke-width:2px
    style SC fill:#e1bee7,stroke:#e1bee7,stroke-width:2px
    style OSL fill:#a3d8b0,stroke:#a3d8b0,stroke-width:2px
    style CM fill:#a3d8b0,stroke:#a3d8b0,stroke-width:2px
    style SME fill:#a3d8b0,stroke:#a3d8b0,stroke-width:2px
    style CONT fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px
    style USERS fill:#f5f5f5,stroke:#9e9e9e,stroke-width:1px
```

---

## Roles and Responsibilities

### European Space Agency (ESA)

**Role:** Mission and repository owner

**Responsibilities:**
- Owns the ESA external Git repository for BIOMASS L2
- Defines high-level requirements for L2 product quality, timeliness, and continuity
- Approves major algorithmic changes and official releases affecting operational processing
- Ensures governance and processes align with ESA policies and Open Science strategy
- Final authority for Tier 2-3 decisions
- Releases approval
- Strategy alignment

**Decision Authority:**
- Tier 2-3 PRs: Final approval required
- Releases: Approval required
- Governance changes: Final decision maker

**Release gate via CODEOWNERS**: any modification to `VERSION` or `CHANGELOG.md` triggers a mandatory review by the designated ESA reviewer defined in the `CODEOWNERS` file. No release can be merged without this approval.

**Branch protection rulesets**: three GitHub rulesets protect the main branches:
- `develop`: 1 approval required, squash-only
- `release`: 2 approvals required, GPG/SSH signed commits, squash-only. Heavy CI is mandatory: every PR targeting `release` is forced to Tier 2 by the CI, and a maintainer must explicitly trigger the workflow with `run_heavy=true` for the `CI gate` to pass.
- `main`: 3 approvals required, GPG/SSH signed commits, squash-only, no admin bypass

The `CI gate` status check is required on all three branches before any merge.

---

### Open Science Lead

**Role:** Open Science strategy and FAIR compliance

**Responsibilities:**
- Designs, implements, and continuously improves the Open Science strategy for BIOMASS L2
- Co-defines governance rules, PR templates, and validation requirements with ESA and scientific experts
- Acts as a bridge between operational constraints and scientific/open-source community expectations
- Reviews Tier 2 PRs for Open Science compliance (transparency, documentation, reproducibility)
- Coordinates community engagement and communication
- Oversees documentation quality and completeness
- Organizes workshops, training, and community events

**Decision Authority:**
- Tier 2 PRs: Approval required (Open Science perspective)
- Documentation standards: Defines and enforces
- Community events: Organizes and coordinates

---

### Core Maintainers

**Role:** Technical review and code quality

**Responsibilities:**
- Enforce contribution and review standards
- Manage protected branches and releases
- Coordinate responses to critical issues and security vulnerabilities
- Technical review for Tier 0-2 PRs
- Merge PRs after all required approvals
- Manage releases and versioning
- Maintain code quality and architecture consistency
- Act as custodians of the long-term health of the codebase

**Decision Authority:**
- Tier 0-2 PRs: Technical approval required
- Code quality: Enforce standards
- Releases: Manage process

**Composition:**
- Small group of individuals from ESA, industrial partners, and/or external experts
- Elevated rights on the repository
- Technical expertise in software development and Earth observation

---

### Scientific Module Experts

**Role:** Scientific validation and methodology

**Responsibilities:**
- Lead scientific development and validation of specific L2 processor components
- Define scientific requirements, validation metrics, and acceptance criteria in their domain
- Review PRs that introduce or modify algorithms in their area of expertise
- Ensure L2 products remain physically consistent and scientifically credible
- Validate methodology and check for scientific regressions
- Provide scientific guidance to contributors

**Decision Authority:**
- Tier 1-2 PRs: Scientific approval required
- Methodology: Validate and approve
- Scientific standards: Define and enforce

**Expertise Domains:**
- **Biomass**: Above-ground biomass estimation algorithms
- **Forest Structure**: Canopy height, structure metrics
- **Uncertainty**: Error propagation and uncertainty quantification


---

### Contributors (Internal and External)

**Role:** Code, documentation, and validation contributions

**Responsibilities:**
- Submit PRs for code, documentation, or validation workflow changes
- Follow contribution guidelines, templates, and coding standards
- Participate in discussions and reviews
- Improve documentation and examples
- Write tests and perform validation
- Engage with the community

**Rights:**
- Submit PRs for any tier
- Participate in discussions
- Attend community meetings
- Access documentation and resources

**Expectations:**
- Follow Code of Conduct
- Respond to review feedback promptly
- Maintain high code quality
- Document changes appropriately

---


---

## Meetings and Communication

For detailed information about meetings, communication channels, and community interactions, please see the [Communication and Meetings Guide](https://biomass-disc.info/docs/communication).

---

## Governance Evolution

### Review Process

The governance framework is reviewed:
- **Annually**: Full review and update
- **As needed**: When significant changes are required
- **Community input**: Solicited during annual review

### Change Process

Changes to governance:
1. Proposal submitted via issue or governance document
2. Discussion in governance meeting
3. Community feedback solicited
4. Final decision by ESA in consultation with governance group
5. Documentation updated
6. Communication to community

---

## Resources

### Documentation

- [Getting Started](https://biomass-disc.info/docs) - Introduction and getting started guide
- [Licensing](https://biomass-disc.info/docs/licensing) - Apache 2.0 license requirements and legal obligations
- [Code of Conduct](https://biomass-disc.info/docs/code-of-conduct) - Community standards and expectations
- [Contributing Guide](https://biomass-disc.info/docs/contributing) - Contribution process and workflows
- [Architecture](https://biomass-disc.info/docs/architecture) - System architecture and design
- [Architecture](https://biomass-disc.info/docs/architecture) - Monorepo layout and `bps-*` modules
- [CI/CD Guide](https://biomass-disc.info/docs/ci-cd-guide) - Pipeline, tier detection, branch protection
- [Release Process](https://biomass-disc.info/docs/release-process) - Release preparation and ESA gate
- [Code Standards](https://biomass-disc.info/docs/code-standards) - Coding conventions and best practices
- [Documentation Standards](https://biomass-disc.info/docs/documentation-standards) - Documentation writing standards and best practices
- [Communication](https://biomass-disc.info/docs/communication) - Communication channels and meeting schedules

### External Resources

- [FAIR Principles](https://www.go-fair.org/fair-principles/)
- [ESA Open Science](https://www.esa.int/Science_Exploration/Space_Science/Open_Science)
- [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/)

---


**Questions about governance?** Open an issue with the `governance` label or contact the Open Science Lead.

---

**Last Updated:** 2026

