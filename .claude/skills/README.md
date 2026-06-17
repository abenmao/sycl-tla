# `.claude/skills/`

Repeatable task playbooks for AI coding agents working on this repo. Each
skill is a self-contained markdown file describing how to accomplish one
specific task — filing a Jira ticket, running a benchmark, applying a code
pattern, etc.

## Available skills

| Skill | What it does |
|---|---|
| [`cutlass-jira/`](cutlass-jira/SKILL.md) | Read / search / create / update / link / transition / comment on tickets in the internal `CUTLASS9` Jira project. |
| [`cutlass-confluence-wiki/`](cutlass-confluence-wiki/SKILL.md) | Read, search, create, update, comment on, or attach to Confluence pages on Intel's internal wiki (`wiki.ith.intel.com`) — defaults to the CUTLASS space / SYCL\*TLA Wiki. |
| [`cutlass-arch-docs/`](cutlass-arch-docs/SKILL.md) | Fetch and read authoritative Intel Xe hardware specs (CRI / Xe3p, JGS, NVL) from Artifactory to verify kernels against the silicon — XeCore layout, systolic shapes, L2, copy engines, FP8/FP4 TOPS, power/perf, hw limits. Index: [`media/docs/cpp/hw/README.md`](../../media/docs/cpp/hw/README.md). |

## How agents use this

- **Claude Code** auto-discovers every `<name>/SKILL.md` here at session
  start (looks for the YAML frontmatter and treats each as an invokable
  skill).
- **Cursor / OpenAI Codex / GitHub Copilot / Aider / Continue** don't
  auto-discover this directory. To use a skill from one of those tools,
  reference the specific `SKILL.md` path in your prompt — every skill is
  plain markdown that any agent can read.
- See [`../../AGENTS.md`](../../AGENTS.md) for the cross-tool routing index.

## Skill file layout

Each skill lives in its own directory:

```
.claude/skills/
└── <skill-name>/
    ├── SKILL.md         # the playbook (required)
    └── ...              # supporting files (optional)
```

`SKILL.md` starts with YAML frontmatter:

```markdown
---
name: <skill-name>
description: <one-line summary, used by Claude Code to decide when to invoke>
---

# <Skill Title>

<body — instructions, commands, examples>
```

## Conventions

- **Never embed secrets.** Skills must not include API keys, PATs,
  passwords, or any credential value. Reference credentials by environment
  variable or `~/.cred`-style local files (which are gitignored).
- **Architecture-neutral.** Where a skill applies to multiple Intel Xe
  architectures (Xe2 / BMG, Xe-HPC / PVC, Xe3p / CRI, JGS, mainline),
  document each path rather than hard-coding one. Use the same vocabulary
  the rest of the repo uses (see `README.md` and
  `media/docs/cpp/cute/10_intel_overview.md`). Ask the user to confirm the
  target if it's not obvious from context.
- **Cite live API discovery.** When a skill describes calling an internal
  API (Jira, Jenkins, etc.), include the discovery commands so the skill
  stays usable when fields or schemas change.
- **AI-assistance disclaimer.** When a skill produces user-facing artifacts
  (Jira tickets, PR comments, …), append a one-line "_created with AI
  assistance_" footer to each artifact.

## Adding a new skill

1. Create `.claude/skills/<skill-name>/SKILL.md` with the frontmatter +
   body shape shown above.
2. Add a row to the table at the top of this README.
3. Add a row to the "By goal" table in [`../../AGENTS.md`](../../AGENTS.md)
   so other tools' users find it.
4. Open a PR. The PR description should explain when the skill is intended
   to be invoked.
