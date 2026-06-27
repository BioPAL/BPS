# Conflict resolution

We recognize that conflicts can arise in any collaborative environment. Our goal is to resolve conflicts fairly, respectfully, and efficiently while maintaining a positive community atmosphere.

## Our Approach

We believe in addressing conflicts early and directly. Our process is designed to be:

- **Fair:** All parties have a chance to be heard
- **Transparent:** Process and outcomes are documented (while respecting privacy)
- **Respectful:** We focus on issues, not personalities
- **Efficient:** We aim to resolve conflicts quickly without unnecessary bureaucracy

## Resolution Process

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

## Principles

When resolving conflicts, we follow these principles:

- **Focus on Issues:** Address the problem, not the person. Critique ideas, not individuals.
- **Seek Understanding:** Try to understand all perspectives before making judgments.
- **Find Common Ground:** Look for solutions that work for everyone involved.
- **Respect Decisions:** Once a decision is made through the proper process, respect it even if you disagree.
- **Maintain Privacy:** Keep sensitive matters private while being transparent about the process.

## Escalation Path

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

## Getting Help

If you're unsure how to handle a conflict or need guidance:

- Review the [Code of Conduct](../contributing/code-of-conduct.md) for behavioral guidelines
- Contact a core maintainer or the Open Science Lead
- Open a private issue or discussion if you prefer anonymity
- Remember: asking for help is always okay

---

**Previous:** [First Developer Meeting](developer-meeting.md) | **Next:** [Getting help](getting-help.md)
