# Agent Routing Index — SYCL*TLA

Cross-tool entry point for AI coding agents. Pointers, not content. Loaded into context once per session per agent in this repo, so it stays short on purpose.

Native-load: Cursor, Codex, Copilot (Coding Agent + CLI), Aider, Continue. Claude Code reads `CLAUDE.md` → `@AGENTS.md`. GitHub Copilot variants additionally read [`.github/copilot-instructions.md`](.github/copilot-instructions.md), which is the authoritative build/CI/PR guide.

## By goal

| Goal | Read first |
|---|---|
| Set up env, build, run tests, ship a CI-safe PR | [`.github/copilot-instructions.md`](.github/copilot-instructions.md) |
| Understand SYCL*TLA code organization | [`media/docs/cpp/code_organization.md`](media/docs/cpp/code_organization.md), [`media/docs/cpp/efficient_gemm.md`](media/docs/cpp/efficient_gemm.md) |
| Understand the Xe architecture redesign (atoms, copies, subgroup model) | [`media/docs/cpp/xe_rearchitecture.md`](media/docs/cpp/xe_rearchitecture.md) |
| Tune a kernel (tile sizing, pipeline stages, prefetch, SLM, reorders) | [`media/docs/cpp/cute/12_intel_performance_guide.md`](media/docs/cpp/cute/12_intel_performance_guide.md) |
| GEMM-specific tuning companion | [`media/docs/cpp/cute/11_intel_gemm_companion.md`](media/docs/cpp/cute/11_intel_gemm_companion.md) |
| Xe arch-tag vocabulary (Xe12 / Xe20 / …) | [`media/docs/cpp/cute/10_intel_overview.md`](media/docs/cpp/cute/10_intel_overview.md) |
| Verify a kernel/change against CRI/Xe3p (or JGS/NVL) HW specs | [`media/docs/cpp/hw/README.md`](media/docs/cpp/hw/README.md) |
| Browse all repeatable team task playbooks | [`.claude/skills/README.md`](.claude/skills/README.md) |
| Read / search / write a CUTLASS9 Jira ticket | [`.claude/skills/cutlass-jira/SKILL.md`](.claude/skills/cutlass-jira/SKILL.md) |
| Fetch / read Intel Xe hardware specs for verification | [`.claude/skills/cutlass-arch-docs/SKILL.md`](.claude/skills/cutlass-arch-docs/SKILL.md) |
| Read or write a Confluence page on the SYCL\*TLA wiki | [`.claude/skills/cutlass-confluence-wiki/SKILL.md`](.claude/skills/cutlass-confluence-wiki/SKILL.md) |
| CI workflows | [`.github/workflows/`](.github/workflows/) |

When adding a new skill or doc, add a row above so all agents find it.
