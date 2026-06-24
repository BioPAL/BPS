# Becoming a maintainer

We value long-term contributors who demonstrate commitment, technical expertise, and community leadership. Becoming a maintainer is a natural progression for contributors who have shown consistent dedication to the project. This section outlines the pathway from contributor to maintainer, inspired by successful open-source projects like Xarray.

### What is a Maintainer?

Maintainers are trusted community members who have demonstrated deep commitment to the project and have been granted elevated permissions on the repository. They play a crucial role in:

- Reviewing and merging pull requests
- Managing releases and versioning
- Setting technical direction and architecture decisions
- Ensuring code quality and consistency
- Mentoring new contributors
- Representing the project in the community

For more details about maintainer responsibilities, see the [Governance documentation](../governance/index.md).

### The Path to Maintainership

Becoming a maintainer is not about a specific number of contributions or a fixed timeline. Instead, it's about demonstrating consistent commitment, technical competence, and alignment with the project's values and goals. The path typically follows these stages:

The following flowchart illustrates the journey from contributor to maintainer:

```{mermaid}
flowchart TD
    Start([New Contributor]) --> Stage1[Stage 1: Active Contributor<br/>3-6 months]
    
    Stage1 --> Contrib1[Regular contributions]
    Stage1 --> Standards[Follow standards]
    Stage1 --> Community[Community engagement]
    
    Contrib1 --> Check1{Quality and<br/>consistency?}
    Standards --> Check1
    Community --> Check1
    
    Check1 -->|Yes| Stage2[Stage 2: Trusted Contributor<br/>6-12 months]
    Check1 -->|No| Stage1
    
    Stage2 --> Review[Code review]
    Stage2 --> Mentor[Mentoring]
    Stage2 --> Complex[Complex issues]
    Stage2 --> Tier12[Tier 1-2 contributions]
    
    Review --> Check2{Leadership and<br/>expertise?}
    Mentor --> Check2
    Complex --> Check2
    Tier12 --> Check2
    
    Check2 -->|Yes| Stage3[Stage 3: Maintainer Candidate]
    Check2 -->|No| Stage2
    
    Stage3 --> Express[Express interest]
    Express --> Eval[Evaluation by maintainers]
    Eval --> Discuss[Internal discussion]
    Discuss --> Nominate{Nomination?}
    
    Nominate -->|Yes| Approve[Steering Council approval]
    Nominate -->|No| Stage2
    
    Approve --> Maintainer([Maintainer])
    
    classDef defaultStyle fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px,color:#333
    class Start,Stage1,Stage2,Stage3,Maintainer,Contrib1,Standards,Community,Check1,Review,Mentor,Complex,Tier12,Check2,Express,Eval,Discuss,Nominate,Approve defaultStyle
    
    style Start fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px
    style Stage1 fill:#e1bee7,stroke:#e1bee7,stroke-width:2px
    style Stage2 fill:#a3d8b0,stroke:#a3d8b0,stroke-width:2px
    style Stage3 fill:#e1bee7,stroke:#e1bee7,stroke-width:2px
    style Maintainer fill:#a3d8b0,stroke:#a3d8b0,stroke-width:3px
    style Check1 fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px
    style Check2 fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px
    style Nominate fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px
```

The following table compares the different stages:

| Criterion | Stage 1: Active Contributor | Stage 2: Trusted Contributor | Stage 3: Maintainer Candidate |
|-----------|----------------------------|-----------------------------|-------------------------------|
| **Typical duration** | 3-6 months | 6-12 months | 2-4 weeks (process) |
| **Contributions** | Regular and quality | Complex and significant | Leadership and responsibilities |
| **Code review** | Submit PRs | Review others' PRs | Advanced review and mentoring |
| **Engagement** | Discussions and questions | Mentoring and triage | Project management |
| **Validation** | Mainly Tier 0 | Tier 1-2 with validation | All tiers |
| **Permissions** | Standard contributor | Trusted contributor | Maintainer candidate |
| **Next step** | Stage 2 | Stage 3 | Maintainer |

#### Stage 1: Active Contributor

**Focus:** Build a track record of quality contributions

**What to do:**
- Make regular, meaningful contributions (code, documentation, tests, reviews)
- Follow all contribution guidelines and coding standards
- Respond promptly to review feedback
- Help improve documentation and examples
- Participate in discussions and answer questions

**What we look for:**
- Consistent contributions over time (not just one large contribution)
- High-quality code that follows our standards
- Good understanding of the codebase and project goals
- Positive interactions with the community

**Timeline:** This stage typically lasts 3-6 months of active contribution, but can vary based on the nature and quality of contributions.

#### Stage 2: Trusted Contributor

**Focus:** Demonstrate leadership and deeper engagement

**What to do:**
- Take ownership of complex issues and features
- Help review pull requests from other contributors
- Mentor new contributors and answer questions
- Propose and implement significant improvements
- Participate actively in technical discussions
- Help triage issues and improve project organization
- Contribute to Tier 1-2 changes with scientific validation

**What we look for:**
- Ability to review code effectively and provide constructive feedback
- Leadership in technical discussions and decision-making
- Mentoring and community-building skills
- Deep understanding of the project's scientific and technical aspects
- Reliability and consistency in contributions

**Timeline:** This stage typically lasts 6-12 months, during which you build trust and demonstrate your commitment.

#### Stage 3: Maintainer Candidate

**Focus:** Express interest and demonstrate readiness

**What to do:**
- Continue all activities from Stage 2
- Express interest in becoming a maintainer to current maintainers or the Open Science Lead
- Take on additional responsibilities when asked
- Help with release management and project maintenance tasks
- Participate in maintainer discussions (as invited)

**What we look for:**
- Proven track record of quality contributions and reviews
- Strong alignment with project values and goals
- Ability to work independently and make sound technical decisions
- Excellent communication and collaboration skills
- Commitment to the project's long-term success

**Process:**
1. **Expression of Interest:** You can express interest directly to current maintainers, the Open Science Lead, or by opening a thread in the [Governance category on GitHub Discussions](https://github.com/BioPAL/BPS/discussions/categories/governance)
2. **Evaluation:** Current maintainers will review your contributions, community engagement, and technical expertise
3. **Discussion:** Maintainers will discuss your candidacy internally
4. **Nomination:** If there's consensus, a maintainer will nominate you
5. **Approval:** The nomination is reviewed by the repository stewards and ESA representatives
6. **Onboarding:** Once approved, you'll receive maintainer permissions and onboarding support

**Timeline:** The evaluation and approval process typically takes 2-4 weeks after expression of interest.

### Key Qualities We Value

While there's no strict checklist, successful maintainers typically demonstrate:

**Technical Excellence:**
- Deep understanding of the codebase and architecture
- Strong software engineering skills
- Knowledge of Earth observation and SAR processing (for scientific aspects)
- Ability to write clear, maintainable code
- Understanding of testing, validation, and quality assurance

**Community Leadership:**
- Positive, respectful communication
- Ability to mentor and guide others
- Constructive feedback and code review skills
- Conflict resolution abilities
- Commitment to inclusivity and diversity

**Project Commitment:**
- Long-term dedication to the project
- Availability for reviews and maintenance tasks
- Alignment with project goals and values
- Understanding of governance and decision-making processes
- Willingness to take on responsibilities

**Scientific Rigor (for scientific aspects):**
- Understanding of validation requirements
- Ability to review scientific changes
- Knowledge of reference datasets and validation methods
- Commitment to reproducibility and transparency

### Maintainer Responsibilities

Once you become a maintainer, you'll take on additional responsibilities:

**Code Review and Merging:**
- Review pull requests across all tiers
- Ensure code quality and standards compliance
- Approve and merge PRs after all requirements are met
- Help resolve conflicts and technical issues

**Project Maintenance:**
- Manage releases and versioning
- Maintain code quality and architecture consistency
- Respond to critical issues and security vulnerabilities
- Manage protected branches and repository settings

**Community Leadership:**
- Mentor new contributors
- Participate in technical and governance meetings
- Represent the project in the community
- Help set technical direction and priorities

**Governance:**
- Participate in maintainer discussions and decisions
- Contribute to governance and policy discussions
- Help ensure compliance with ESA policies and Open Science principles

### How to Get Started

If you're interested in becoming a maintainer:

1. **Start Contributing:** Focus on making quality contributions and building your track record
2. **Engage with the Community:** Participate in discussions, help others, and build relationships
3. **Take on Challenges:** Volunteer for complex issues and significant features
4. **Review Code:** Start reviewing other contributors' pull requests
5. **Express Interest:** When you feel ready, express your interest to current maintainers

**Remember:** There's no rush. Focus on making quality contributions and building trust with the community. Maintainership will come naturally as you demonstrate your commitment and capabilities.

### Questions?

If you have questions about the maintainer path or want to discuss your journey:

- Start a thread in the [Governance category on GitHub Discussions](https://github.com/BioPAL/BPS/discussions/categories/governance)
- Contact current maintainers directly
- Talk to the Open Science Lead
- Ask during Community Meetings or Office Hours


---

**Previous:** [Templates and checklists](templates-and-checklists.md) | **Next:** [Release process](release-process.md)
