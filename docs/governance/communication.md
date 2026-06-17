# Welcome to BioPAL Communication

Welcome to the BioPAL community! This guide will help you understand how we communicate, collaborate, and work together. Whether you're a new contributor or a long-time maintainer, this document will help you find the right channel for your questions, ideas, and discussions.

We believe that clear communication is the foundation of a healthy open-source community. This guide covers everything from where to ask questions to how we handle meetings and resolve conflicts.

---

## Quick Start: Where Should I Go?

**I have a question or need help:**
- General questions and community discussions → [GitHub Discussions](https://github.com/BioPAL/BPS/discussions)
- Bug to report → [GitHub Issues](https://github.com/BioPAL/BioPAL/issues)
- Need quick help → Office Hours (available later)

**I want to contribute:**
- Start with the [Contributing Guide](https://biomass-disc.info/docs/contributing)
- Join our Technical Meetings (bi-monthly)
- Attend an Onboarding Workshop (monthly)

**I have a conflict or concern:**
- Review our [Conflict Resolution](#conflict-resolution) process
- Check the [Code of Conduct](https://biomass-disc.info/docs/code-of-conduct)

---

## Communication Channels

We use multiple channels to ensure everyone can participate in the way that works best for them. Here's when to use each one:

The following flowchart helps you choose the right communication channel:

```{mermaid}
flowchart TD
    Start([I need to communicate]) --> Type{Type of communication?}
    
    Type -->|General question| General{Urgent?}
    Type -->|Bug to report| Bug[GitHub Issues]
    Type -->|Feature idea| Feature[GitHub Issues]
    Type -->|Discussion| Discussion[Forum<br/>Available early 2026]
    Type -->|Conflict| Conflict[Conflict Resolution Process]
    Type -->|Security| Security[Security Email]
    
    General -->|Yes| OfficeHours[Office Hours<br/>Weekly]
    General -->|No| Discussion
    
    Bug --> IssueForm[Fill Issue template]
    Feature --> IssueForm
    IssueForm --> Submit[Submit Issue]
    
    Discussion --> Search{Search existing<br/>discussions?}
    Search -->|Found| Reply[Reply to discussion]
    Search -->|Not found| New[Create new discussion]
    
    OfficeHours --> Join[Join session]
    
    Conflict --> ConflictProcess[See Conflict<br/>Resolution section]
    
    Security --> SecurityEmail[Security email]
    
    classDef defaultStyle fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px,color:#333
    class Start,Type,General,Bug,Feature,Discussion,Conflict,Security,OfficeHours,IssueForm,Submit,Search,Reply,New,Join,ConflictProcess,SecurityEmail defaultStyle
    
    style Start fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px
    style Bug fill:#ef9a9a,stroke:#e57373,stroke-width:2px
    style Feature fill:#e1bee7,stroke:#e1bee7,stroke-width:2px
    style Discussion fill:#a3d8b0,stroke:#a3d8b0,stroke-width:2px
    style OfficeHours fill:#e1bee7,stroke:#e1bee7,stroke-width:2px
    style Security fill:#ef9a9a,stroke:#e57373,stroke-width:2px
```

### GitHub Issues

**Best for:** Bug reports, feature requests, technical questions, and public discussions

**[Open an Issue](https://github.com/BioPAL/BioPAL/issues)**

Use GitHub Issues when you want to:
- Report a bug or unexpected behavior
- Request a new feature or enhancement
- Ask technical questions that need tracking
- Start a public discussion that requires follow-up

**Tips:**
- Use clear, descriptive titles
- Include relevant labels if you have permission
- Reference related issues or pull requests
- Be patient and respectful in discussions

### Forum

**Best for:** General questions, community discussions, best practices, and non-urgent topics

**Status:** Forum website is under development and should be available early 2026

The BioPAL community forum will be the primary place for:
- Asking general questions about the project
- Sharing ideas or best practices
- Discussing non-urgent topics
- Connecting with other community members

****On GitHub Discussions:****
- Use GitHub Issues for questions and discussions
- Join Office Hours for real-time discussions
- Check the Communication section for updates on forum availability

****Tips for posting on Discussions:****
- Search existing discussions before posting
- Use appropriate categories (Q&A, Ideas, General, etc.)
- Help others by answering questions when you can

### Office Hours

**Best for:** Quick clarifications, one-on-one support, and getting unstuck

**Schedule:** Weekly, 1 hour sessions

Office Hours are open sessions where maintainers and experienced contributors are available to help. This is perfect for:
- Quick questions that don't need a formal issue
- Getting help with setup or configuration
- Discussing contribution ideas before opening a PR
- One-on-one support for new contributors

**How to join:**
- Check GitHub Issues for the weekly announcement (see [GitHub Discussions](https://github.com/BioPAL/BPS/discussions))
- Join the video call or chat room
- No appointment needed - just drop in!

### Email / Mailing List

**Best for:** Important announcements, in-depth technical discussions, security issues, and governance matters

Use email for:
- Important project announcements
- Security-related concerns (use the security email if available)
- In-depth technical discussions that benefit from longer-form communication
- Governance and policy matters

**Note:** Email addresses will be provided in the Contact section below.

---

## Meetings and Events

We organize several types of meetings to keep our community connected and moving forward. All meetings are open to the community unless otherwise specified.

### Technical Meetings

**Who:** Maintainers and active contributors  
**When:** Twice per month (bi-monthly)  
**Duration:** 1 hour  
**Format:** Video call with public meeting notes

**What we discuss:**
- Review of Tier 0-1 Pull Requests
- Technical discussions and architecture decisions
- Short-term planning and priorities
- Code quality improvements
- Quick wins and blockers

**Why attend:**
- Stay up-to-date with project developments
- Influence technical decisions
- Get your questions answered quickly
- Connect with other contributors

**How to join:**
- Meeting invitations are shared via GitHub Issues (see [GitHub Discussions](https://github.com/BioPAL/BPS/discussions))
- Meeting notes are published publicly after each session
- All are welcome, but active contributors are especially encouraged to attend

### Governance Meetings

**Who:** Steering Committee (ESA, Open Science Lead, Core Maintainers, Scientific Experts)  
**When:** Once per quarter  
**Duration:** 1.5 hours  
**Format:** Video call with public meeting notes and decisions

**What we discuss:**
- Review of Tier 2 Pull Requests
- Strategic decisions and roadmap
- Policy updates and governance changes
- Resource allocation and priorities
- Long-term vision and planning

**Why attend:**
- Understand the strategic direction of the project
- See how major decisions are made
- Provide input on governance matters
- Learn about upcoming priorities

**How to join:**
- Open to all community members as observers
- Meeting invitations are shared via GitHub Issues (see [GitHub Discussions](https://github.com/BioPAL/BPS/discussions))
- All decisions and notes are published publicly

### Community Meetings

**Who:** All contributors and users  
**When:** Once per quarter  
**Duration:** 1 hour  
**Format:** Video call with public meeting notes and recordings

**What we discuss:**
- Feature presentations and demos
- Q&A sessions with maintainers
- Feedback collection from users
- Project announcements and updates
- Community highlights and recognition

**Why attend:**
- Learn about new features and improvements
- Get your questions answered directly
- Share your feedback and ideas
- Connect with the broader BioPAL community

**How to join:**
- Meeting invitations are shared widely via GitHub Issues and email (see [GitHub Discussions](https://github.com/BioPAL/BPS/discussions))
- Recordings are made available for those who can't attend
- All are welcome and encouraged to participate

### Workshops and Training

We regularly organize educational events to help community members learn and grow:

**Onboarding Workshops** (Monthly, 2 hours)
- Perfect for new contributors
- Learn about the project structure, development workflow, and contribution process
- Hands-on exercises and Q&A

**Technical Training** (Quarterly)
- Deep dives into specific technical topics
- Advanced development techniques
- Best practices and patterns

**Hackathons** (Annual)
- Intensive coding sessions
- Collaborative problem-solving
- Great for building relationships and making progress on challenging issues

**Intercomparison Exercises** (Regular)
- Scientific validation activities
- Data analysis and comparison
- Quality assurance and testing

**How to find out about events:**
- Check GitHub Issues for announcements (see [GitHub Discussions](https://github.com/BioPAL/BPS/discussions))
- Subscribe to project updates
- Follow the project on social media (if available)

---

## Conflict Resolution

We recognize that conflicts can arise in any collaborative environment. Our goal is to resolve conflicts fairly, respectfully, and efficiently while maintaining a positive community atmosphere.

### Our Approach

We believe in addressing conflicts early and directly. Our process is designed to be:

- **Fair:** All parties have a chance to be heard
- **Transparent:** Process and outcomes are documented (while respecting privacy)
- **Respectful:** We focus on issues, not personalities
- **Efficient:** We aim to resolve conflicts quickly without unnecessary bureaucracy

### Resolution Process

The following flowchart illustrates the conflict resolution process:

```{mermaid}
flowchart TD
    Conflict([Conflict identified]) --> Step1[Step 1: Direct Communication]
    
    Step1 --> Try1{Resolution<br/>attempt?}
    Try1 -->|Success| Resolved([Conflict resolved])
    Try1 -->|Failure| Step2[Step 2: Mediation]
    
    Step2 --> Mediator{Conflict type?}
    Mediator -->|Contributor ↔ Contributor| CM1[Core Maintainer<br/>as mediator]
    Mediator -->|Contributor ↔ Maintainer| OSL1[Open Science Lead<br/>as mediator]
    Mediator -->|Maintainer ↔ Maintainer| OSL2[Open Science Lead<br/>as mediator]
    
    CM1 --> Try2{Resolution?}
    OSL1 --> Try2
    OSL2 --> Try2
    
    Try2 -->|Success| Resolved
    Try2 -->|Failure| Step3[Step 3: Escalation]
    
    Step3 --> Escalate[Escalate to<br/>Open Science Lead or ESA]
    Escalate --> Governance[Review by<br/>Governance Group]
    Governance --> Decision[Final decision]
    Decision --> Resolved
    
    Resolved --> Document[Document conflict<br/>and resolution]
    
    classDef defaultStyle fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px,color:#333
    class Conflict,Step1,Step2,Step3,Resolved,Try1,Mediator,CM1,OSL1,OSL2,Try2,Escalate,Governance,Decision,Document defaultStyle
    
    style Conflict fill:#ef9a9a,stroke:#e57373,stroke-width:2px
    style Step1 fill:#e1bee7,stroke:#e1bee7,stroke-width:2px
    style Step2 fill:#a3d8b0,stroke:#a3d8b0,stroke-width:2px
    style Step3 fill:#e1bee7,stroke:#e1bee7,stroke-width:2px
    style Resolved fill:#a3d8b0,stroke:#a3d8b0,stroke-width:3px
    style Try1 fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px
    style Try2 fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px
```

**Step 1: Direct Communication**

Start by trying to resolve the conflict directly with the involved parties. Often, a respectful conversation can clear up misunderstandings.

- Use private channels (direct messages, email) for sensitive matters
- Focus on the specific issue, not personal attributes
- Listen actively and try to understand different perspectives
- Look for common ground and mutually acceptable solutions

**Step 2: Mediation**

If direct communication doesn't resolve the issue, involve a neutral party:

- For contributor-to-contributor conflicts: Contact a core maintainer
- For contributor-to-maintainer conflicts: Contact the Open Science Lead
- For maintainer-to-maintainer conflicts: Contact the Open Science Lead or ESA representative

The mediator will:
- Listen to all parties
- Help clarify the issues
- Facilitate a constructive discussion
- Work toward a mutually acceptable solution

**Step 3: Escalation**

For serious conflicts or when mediation doesn't resolve the issue:

- Escalate to the Open Science Lead or ESA representative
- The governance group will review the situation
- A formal decision will be made and communicated to all parties
- The resolution will be documented (respecting privacy)

**Step 4: Documentation**

All conflicts and their resolutions are documented (with appropriate privacy considerations) to:
- Ensure consistency in how conflicts are handled
- Learn from past experiences
- Maintain transparency in the process

### Principles

When resolving conflicts, we follow these principles:

- **Focus on Issues:** Address the problem, not the person. Critique ideas, not individuals.
- **Seek Understanding:** Try to understand all perspectives before making judgments.
- **Find Common Ground:** Look for solutions that work for everyone involved.
- **Respect Decisions:** Once a decision is made through the proper process, respect it even if you disagree.
- **Maintain Privacy:** Keep sensitive matters private while being transparent about the process.

### Escalation Path

Here's the typical escalation path for different types of conflicts:

1. **Contributor ↔ Contributor**
   - Start with direct discussion
   - If needed, involve a core maintainer as mediator

2. **Contributor ↔ Maintainer**
   - Start with direct discussion
   - If needed, involve another core maintainer or Open Science Lead as mediator

3. **Maintainer ↔ Maintainer**
   - Start with direct discussion
   - If needed, involve the Open Science Lead as mediator
   - Escalate to ESA if necessary

4. **Governance Issues**
   - Discuss in Governance Meetings
   - Final decision by ESA in consultation with governance group

### Getting Help

If you're unsure how to handle a conflict or need guidance:

- Review the [Code of Conduct](https://biomass-disc.info/docs/code-of-conduct) for behavioral guidelines
- Contact a core maintainer or the Open Science Lead
- Open a private issue or discussion if you prefer anonymity
- Remember: asking for help is always okay

---

## Code of Conduct

All community members are expected to follow our [Code of Conduct](https://biomass-disc.info/docs/code-of-conduct). The Code of Conduct ensures a welcoming, inclusive, and respectful environment for everyone.

**Key points:**
- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

**Reporting violations:**
If you experience or witness behavior that violates the Code of Conduct, please refer to our [Code of Conduct Reporting Guide](https://biomass-disc.info/docs/code-of-conduct#reporting-guide). We take all reports seriously and handle them confidentially.

---

## Getting Help

**Still not sure where to go?**

- **General questions:** Forum (available early 2026) - Use GitHub Issues until then
- **Specific issues:** [GitHub Issues](https://github.com/BioPAL/BioPAL/issues)
- **Quick help:** Office Hours (weekly)
- **Communication questions:** Ask in the [Governance category on GitHub Discussions](https://github.com/BioPAL/BPS/discussions/categories/governance)
- **Code of Conduct concerns:** See the [Reporting Guide](https://biomass-disc.info/docs/code-of-conduct#reporting-guide)

**Contact Information:**

- **Open Science Lead:** [Contact Information]
- **Core Maintainers:** [Contact Information]
- **ESA Representative:** [Contact Information]

---

## Contributing to This Guide

This communication guide is a living document. If you have suggestions for improvement, please:

1. Open an issue or discussion with your ideas
2. Propose changes via a pull request
3. Share feedback during Community Meetings

We're always looking to improve how we communicate and collaborate!

---

**Last Updated:** 2025

