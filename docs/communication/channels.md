# Communication channels

We use multiple channels to ensure everyone can participate in the way that works best for them. Here's when to use each one:

The following flowchart helps you choose the right communication channel:

```{mermaid}
flowchart TD
    Start([I need to communicate]) --> Type{Type of communication?}
    
    Type -->|General question| General{Urgent?}
    Type -->|Bug to report| Bug[GitHub Issues]
    Type -->|Feature idea| Feature[GitHub Issues]
    Type -->|Discussion| Discussion[GitHub Discussions]
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

## GitHub Issues

**Best for:** Actionable items that need triage, tracking, and implementation

**[Open an Issue](https://github.com/BioPAL/BPS/issues/new/choose)**

Use GitHub Issues when you want to:
- Report a bug or unexpected behavior
- Request a new feature or enhancement
- Propose a scientific algorithm or methodological change
- Report a documentation gap or security concern

Open issues are reserved for actionable work. Questions, brainstorming,
and open-ended conversations belong on GitHub Discussions.

**Tips:**
- Pick the matching issue template
- Use clear, descriptive titles
- Reference related issues or pull requests
- Wait for an approval label before starting implementation

## GitHub Discussions

**Best for:** General questions, ideas, scientific debate, and community conversation

**[Open a Discussion](https://github.com/BioPAL/BPS/discussions)**

Use GitHub Discussions when you want to:
- Ask questions about usage, installation, or the processing chain
- Brainstorm an idea before opening a tracking issue
- Discuss algorithms, validation, or methodology
- Share usage stories, papers, or conference talks

Six categories help route the conversation:

| Category | When to use |
|---|---|
| 📢 [Announcements](https://github.com/BioPAL/BPS/discussions/categories/announcements) | Project announcements, releases, governance decisions. Posts restricted to maintainers. |
| ❓ [Q&A](https://github.com/BioPAL/BPS/discussions/categories/q-a) | Usage, installation, API, processing chain, data formats. Mark the helpful reply as the answer. |
| 💡 [Ideas](https://github.com/BioPAL/BPS/discussions/categories/ideas) | Brainstorm a feature or a change before opening a tracking issue. |
| 🔬 [Scientific discussions](https://github.com/BioPAL/BPS/discussions/categories/scientific-discussions) | Algorithms, validation, methodology, ATBD interpretations, references. |
| 🏛️ [Governance](https://github.com/BioPAL/BPS/discussions/categories/governance) | Project governance, maintainer paths, policy questions. |
| 👋 [Show and tell](https://github.com/BioPAL/BPS/discussions/categories/show-and-tell) | Introductions, usage stories, papers, conference talks. |

**Tips:**
- Search existing discussions before posting
- Pick the category that best matches your topic
- Mark a helpful reply as the answer in Q&A
- Open a tracking issue once the conversation becomes actionable

## Office Hours

```{warning}
**Office Hours are not scheduled yet.** The format below describes what we plan to offer in the future. Until then, use [Q&A on GitHub Discussions](https://github.com/BioPAL/BPS/discussions/categories/q-a) for quick questions.
```

**Best for:** Quick clarifications, one-on-one support, and getting unstuck

**Schedule:** Weekly, 1 hour sessions (planned)

Office Hours are open sessions where maintainers and experienced contributors are available to help. This is perfect for:
- Quick questions that don't need a formal issue
- Getting help with setup or configuration
- Discussing contribution ideas before opening a PR
- One-on-one support for new contributors

**How to join:**
- Check [Announcements on GitHub Discussions](https://github.com/BioPAL/BPS/discussions/categories/announcements) for the weekly schedule
- Join the video call or chat room
- No appointment needed - just drop in!

## Email / Mailing List

**Best for:** Important announcements, in-depth technical discussions, security issues, and governance matters

Use email for:
- Important project announcements
- Security-related concerns (use the security email if available)
- In-depth technical discussions that benefit from longer-form communication
- Governance and policy matters

**Note:** Email addresses are listed on the [Getting help](getting-help.md) page.

---

**Previous:** [Communication](index.md) | **Next:** [Meetings and events](meetings.md)
