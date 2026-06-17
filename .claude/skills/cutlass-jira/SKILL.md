---
name: cutlass-jira
description: Use when reading, searching (JQL), creating, updating, linking, transitioning, or commenting on Jira tickets in the CUTLASS9 project on Intel's internal Jira (jira.devtools.intel.com). Covers all SYCL*TLA architecture targets — Xe2 / BMG, Xe-HPC / PVC, Xe3p / CRI, JGS, and mainline.
---

# CUTLASS9 Jira

Read, search, create, update, link, transition, and comment on Jira tickets in the **CUTLASS9** project on Intel's internal Jira (`https://jira.devtools.intel.com`). The web project URL uses `CUTLASS9` as the key — the displayed project *name* is "CUTLASS". Companion to [`cutlass-confluence-wiki/SKILL.md`](../cutlass-confluence-wiki/SKILL.md) — tracking goes here, design / RCA / how-to content goes on the wiki.

## Credentials and safety

Each contributor stores their own Personal Access Token (PAT) on their workstation in `~/.cred` (mode `600`):

```
CUTLASS_JIRA_URL=https://jira.devtools.intel.com
CUTLASS_JIRA_PAT=<your-personal-access-token>
```

`~/.cred` is **outside the repository** and never committed.

**Hard rules:**
- **Never embed a PAT in this skill, in any file in the repository, in commit messages, or in PR descriptions.** The skill always references the token as `$CUTLASS_JIRA_PAT` — a shell variable populated at runtime.
- **Never write a PAT into a config that gets checked in.**
- If a PAT ever appears in git history (a misclick, a paste into a tracked file, …): rotate it immediately at `$CUTLASS_JIRA_URL/secure/ViewProfile.jspa`, then scrub history with `git filter-repo` and force-push.

**Always load credentials first** in your shell session (or before the first `curl` command that uses `$CUTLASS_JIRA_*`):

    set -a; source ~/.cred; set +a

### TLS verification

The default in every example below is `curl -s` — silent, with normal TLS verification. Disabling TLS verification (`-k` / `--insecure`) makes man-in-the-middle attacks undetectable, so do not use `-k` casually.

If `curl -s` against `$CUTLASS_JIRA_URL` fails with a TLS error on your machine, the cause is almost always a missing Intel internal CA chain. The right fix is to install the Intel CA bundle (your IT distribution channel) so verification succeeds. Only as a last resort — when you are certain the connection cannot be intercepted (e.g. on Intel's internal network) and you cannot install the CA — fall back to `curl -sk` (or `--insecure`). Document the fallback in the calling script and treat it as a temporary workaround, not a default.

If a call returns `401 Unauthorized`, the PAT has expired or been revoked. Regenerate at `$CUTLASS_JIRA_URL/secure/ViewProfile.jspa` → Personal Access Tokens, then update `~/.cred`.

## Project basics

| Field | Value |
|-------|-------|
| Project key | `CUTLASS9` |
| Project ID | `122709` |
| Project name | CUTLASS |
| Issue types | `Bug`, `Task`, `Story`, `Epic`, `Feature`, `Sighting`, `Sub-task`, `Test`, `Requirement`, `Initiative`, `Checklist`, `SDLE Task` |

## Required & common fields

The `createmeta` endpoint is restricted on this Jira (returns 404). Field IDs below were discovered by inspecting existing CUTLASS9 issues. Always include `project`, `issuetype`, `summary`, `description`. The other fields are *screen-conditional* — what's accepted depends on the issue type. Sending a field that isn't on the screen returns:

```
400 — "Field 'customfield_XXXXX' cannot be set. It is not on the appropriate screen, or unknown."
```

Drop the offending field(s) and retry — the schema is per-issuetype, not per-project.

### Always (every issue type)

| Field | JSON key | Example value |
|-------|----------|---------------|
| Project key | `project.key` | `"CUTLASS9"` |
| Issue type | `issuetype.name` | `"Bug"`, `"Task"`, `"Story"`, … |
| Summary | `summary` | Short title (≤255 chars) |
| Description | `description` | Jira wiki markup (see reference below) |
| Priority | `priority.id` | `"4"` (P3-Medium) |
| Components | `components` | `[{"id": "281607"}]` |
| Affects version | `versions` | `[{"id": "318838"}]` |
| Fix version | `fixVersions` | `[{"id": "318838"}]` |
| How Found | `customfield_18102.value` | e.g. `"Code Review"`, `"Testing"`, `"Customer"` |
| Submitter Role | `customfield_20837.value` | e.g. `"Developer"`, `"QA"` |
| Exposure | `customfield_13805.value` | e.g. `"2-High"`, `"3-Medium"`, `"4-Low"` |
| Operating System/s | `customfield_11119` | `[{"value": "All"}]` |
| Doc Needed | `customfield_14611.value` | `"Yes"`, `"No"`, `"Unknown"` |
| Additional Requirements | `customfield_40102` | `[{"value": "None"}]` |

### Bug-only (do NOT include on Task / Story / Epic — they 400)

| Field | JSON key | Example value |
|-------|----------|---------------|
| Is this a Regression? | `customfield_20821.value` | `"Yes"` or `"No"` |
| Suspected Security Defect | `customfield_38105.value` | `"No"` (default) |
| What happened | `customfield_43304` | Free text |
| What should have happened | `customfield_43303` | Free text |
| Environment Found | `customfield_22600.value` | Environment label |

> If a field's allowed values aren't listed here, look one up from a sibling ticket: `curl -s -H "Authorization: Bearer $CUTLASS_JIRA_PAT" "$CUTLASS_JIRA_URL/rest/api/2/issue/CUTLASS9-XXX?expand=names" | jq '.names, .fields.customfield_XXXXX'`.

## Priorities (CUTLASS9 uses these)

| Priority | ID |
|----------|----|
| P1-Stopper | 2 |
| P2-High | 3 |
| P3-Medium | 4 |
| P4-Low | 5 |
| Undecided | 6 |

The Jira instance also exposes other priority schemes (Critical, High, etc.) — do not use them on CUTLASS9 unless you've confirmed against an existing ticket.

## Components

| Component | ID |
|-----------|-----|
| Benchmark | 261304 |
| Block FP8 | 281620 |
| Cross Access | 281617 |
| CUTE API | 281607 |
| Epilogure | 281606 |
| EVT Backend | 281610 |
| Example | 281609 |
| Flash Attention | 262031 |
| GEMM | 281604 |
| Grouped GEMM | 281605 |
| IGC | 281612 |
| Performance | 281621 |
| Python API | 281618 |
| Quantization API | 281616 |
| Reorder API | 281619 |
| Simulator | 281608 |
| SLM API | 281703 |
| SYCL | 281611 |
| unspecified | 261114 |
| xe4 | 264000 |

> Note: `Epilogure` is the literal component name as registered on the
> Jira project (id `281606`) — an upstream typo of *Epilogue* that the
> Jira REST API matches verbatim. Do not "fix" it in this table; the
> misspelling is what callers must send.

**Component vs Fix Version (commonly confused):**
- *Component* = the part of the codebase the issue lives in (e.g. `GEMM`, `CUTE API`, `Flash Attention`).
- *Fix Version* = the release/branch the fix is targeted at (e.g. `v0.9.2-cri`).

These are independent. Always pick the *functional* component(s) — what part of the code does the issue touch.

### Architecture targeting

The same repo (SYCL*TLA / cutlass-sycl) covers multiple Intel GPU architectures. The repo vocabulary used elsewhere (see `README.md` and `media/docs/cpp/cute/10_intel_overview.md`):

| Hardware | Generation label | CUTLASS arch tag | DPCPP target |
|---|---|---|---|
| Intel Data Center GPU Max (Ponte Vecchio) | **Xe-HPC** | `Xe12` | `intel_gpu_pvc` |
| Intel Arc B580 / Battlemage | **Xe2** (BMG) | `Xe20` | `intel_gpu_bmg_g21` / `g31` |
| Crescent Island (next) | **Xe3p** (CRI) | (in-progress) | `intel_gpu_cri` |
| Future arch | **Xe4** | (future) | (future) |

**Architecture-named Jira component:** only `xe4` (id `264000`) exists. It refers to the *future* Xe4 architecture, **not** to BMG and **not** as a generic "Intel GPU" tag. Don't use it for BMG, PVC, or CRI work.

**Architectures that have NO dedicated component** (BMG, PVC, CRI, JGS) are signaled instead via:
- A summary prefix — `[BMG]`, `[PVC]`, `[CRI]`, `[JGS]`, etc.
- A suffixed fix version — `v0.9.2-cri` for CRI work, `v0.9.2-jgs` for JGS, plain `v0.9.2` for mainline / cross-arch / BMG / PVC.

**Decision flow when filing a ticket:**

1. Determine which architecture the work targets:
   - **CRI / Xe3p** → use only functional components. Add `[CRI]` summary prefix and `-cri` fix-version suffix.
   - **JGS** → use only functional components. Add `[JGS]` prefix and `-jgs` suffix.
   - **BMG / Xe2 (Xe20)** → use only functional components. Add `[BMG]` prefix and plain fix version.
   - **PVC / Xe-HPC (Xe12)** → use only functional components. Add `[PVC]` prefix and plain fix version.
   - **Future Xe4** → component **may** include `xe4` plus the functional one(s).
   - **Mainline / cross-arch / unclear** → use only functional components, no architecture prefix.

2. **If the architecture is not obvious from context — ASK the user before posting.** Wrong architecture = wrong component + wrong fix version, which mis-routes the ticket. Print the proposed payload and ask which arch this is for.

3. It's fine to leave `components` empty if no functional one fits — many real tickets do.

## Versions (most recent)

| Version | ID | Released |
|---------|-----|----------|
| v0.9.2-cri | 318838 | no |
| v0.9.2 | 318837 | no |
| v0.9.2-jgs | 320006 | no |
| v0.9.1-cri | 317512 | no |
| v0.9.1 | 317511 | no |
| v0.9.1-jgs | 320007 | no |
| v0.9-cri | 312312 | yes |
| v0.9 | 312705 | yes |
| v0.9-jgs | 313917 | yes |
| v0.8-cri | 306614 | yes |
| v0.8 | 306622 | yes |
| Backlog-cri | 297726 | no |

The version table above can drift as the project releases. Look up live values:

```bash
curl -s -H "Authorization: Bearer $CUTLASS_JIRA_PAT" \
  "$CUTLASS_JIRA_URL/rest/api/2/project/CUTLASS9/versions" \
  | jq 'sort_by(.id | tonumber) | reverse | .[] | {name, id, released}'
```

## Workflow

1. **Confirm the architecture target if unclear.** The same repo covers BMG / PVC / CRI / JGS / future Xe4 / mainline, and they take different components, fix-version suffixes, and summary prefixes (see "Architecture targeting" above). Before drafting a payload, if the target architecture isn't obvious from the user's request or the working directory, ASK which one — wrong architecture mis-routes the ticket.
2. **User approval required.** ALWAYS print the full JSON payload and ask for explicit confirmation before creating, updating, linking, transitioning, or commenting on any ticket. Never write to Jira silently.
3. **AI-generated disclaimer.** Every description and comment created via this skill MUST end with one of:
   - `_This ticket was created with AI assistance._`
   - `_This comment was created with AI assistance._`
4. **Verify keys before linking.** Before creating a link between two tickets, fetch each key's summary to confirm it exists and is the right one.
5. **Report results.** After creating a ticket, return the new key and a browse URL: `$CUTLASS_JIRA_URL/browse/CUTLASS9-XXXXX`.

## Creating a Bug — full examples

Two examples below — one for CRI / Xe3p (uses `-cri` fix version, no `xe4` component, `[CRI]` prefix) and one for BMG / Xe2 (plain fix version, no `xe4` component, `[BMG]` prefix). Pick the pattern that matches the architecture target (see "Architecture targeting" above).

### Example A — CRI / Xe3p bug

```bash
set -a; source ~/.cred; set +a

curl -s -X POST \
  -H "Authorization: Bearer $CUTLASS_JIRA_PAT" \
  -H "Content-Type: application/json" \
  "$CUTLASS_JIRA_URL/rest/api/2/issue" \
  -d '{
    "fields": {
      "project": {"key": "CUTLASS9"},
      "issuetype": {"name": "Bug"},
      "summary": "[CRI] GEMM kernel returns wrong result with mixed dtypes",
      "description": "h3. Problem\nGEMM with bf16 A and fp16 B returns incorrect tile values on the CRI/Xe3p simulator.\n\nh3. Steps to Reproduce\n# Build with cutlass v0.9.2-cri\n# Run {{cri_gemm_mixed_dtype}} test\n# Compare against reference\n\nh3. Expected\nMatches reference within 1e-3 tolerance.\n\nh3. Actual\n{code}\nmax abs error: 0.42\n{code}\n\n_This ticket was created with AI assistance._",
      "priority": {"id": "3"},
      "components": [{"id": "281604"}],
      "versions": [{"id": "318838"}],
      "fixVersions": [{"id": "318838"}],
      "customfield_20821": {"value": "No"},
      "customfield_18102": {"value": "Testing"},
      "customfield_20837": {"value": "Developer"},
      "customfield_13805": {"value": "2-High"},
      "customfield_11119": [{"value": "All"}],
      "customfield_14611": {"value": "No"},
      "customfield_38105": {"value": "No"},
      "customfield_40102": [{"value": "None"}]
    }
  }'
```

### Example B — BMG / Xe2 bug

Differences from Example A:
- Summary prefix `[BMG]` instead of `[CRI]`
- `components` lists only the functional component (no `xe4` — that component is reserved for the future Xe4 architecture, not BMG)
- Fix/affects versions use the **plain** id (no `-cri` suffix) — e.g. `v0.9.2` is `318837`

```bash
curl -s -X POST \
  -H "Authorization: Bearer $CUTLASS_JIRA_PAT" \
  -H "Content-Type: application/json" \
  "$CUTLASS_JIRA_URL/rest/api/2/issue" \
  -d '{
    "fields": {
      "project": {"key": "CUTLASS9"},
      "issuetype": {"name": "Bug"},
      "summary": "[BMG] FlashAttention prefill regression with FP8 KV",
      "description": "h3. Problem\nFA prefill with FP8 KV cache produces wrong logits on BMG (Xe2 / Xe20).\n\n_This ticket was created with AI assistance._",
      "priority": {"id": "3"},
      "components": [{"id": "262031"}],
      "versions": [{"id": "318837"}],
      "fixVersions": [{"id": "318837"}],
      "customfield_20821": {"value": "No"},
      "customfield_18102": {"value": "Testing"},
      "customfield_20837": {"value": "Developer"},
      "customfield_13805": {"value": "2-High"},
      "customfield_11119": [{"value": "All"}],
      "customfield_14611": {"value": "No"},
      "customfield_38105": {"value": "No"},
      "customfield_40102": [{"value": "None"}]
    }
  }'
```

The response is `{"id": "...", "key": "CUTLASS9-XXXX", "self": "..."}`.

## Updating a ticket

```bash
curl -s -X PUT \
  -H "Authorization: Bearer $CUTLASS_JIRA_PAT" \
  -H "Content-Type: application/json" \
  "$CUTLASS_JIRA_URL/rest/api/2/issue/CUTLASS9-1234" \
  -d '{
    "fields": {
      "priority": {"id": "2"},
      "components": [{"id": "264000"}]
    }
  }'
```

## Adding a comment

```bash
curl -s -X POST \
  -H "Authorization: Bearer $CUTLASS_JIRA_PAT" \
  -H "Content-Type: application/json" \
  "$CUTLASS_JIRA_URL/rest/api/2/issue/CUTLASS9-1234/comment" \
  -d '{
    "body": "Reproduced on CRI sim build 2026-05-27. Root cause appears to be in tile epilogue.\n\n_This comment was created with AI assistance._"
  }'
```

## Linking issues

```bash
curl -s -X POST \
  -H "Authorization: Bearer $CUTLASS_JIRA_PAT" \
  -H "Content-Type: application/json" \
  "$CUTLASS_JIRA_URL/rest/api/2/issueLink" \
  -d '{
    "type": {"name": "Blocks"},
    "inwardIssue": {"key": "CUTLASS9-1234"},
    "outwardIssue": {"key": "CUTLASS9-5678"}
  }'
```

Common link types on this Jira: `Blocks`, `Relates`, `Cloners`, `Duplicate`, `Depends on`, `Problem/Incident` (for "causes"/"is caused by"), `Regression`, `Fixes`. Full list:

```bash
curl -s -H "Authorization: Bearer $CUTLASS_JIRA_PAT" \
  "$CUTLASS_JIRA_URL/rest/api/2/issueLinkType" | jq '.issueLinkTypes[] | {name, inward, outward}'
```

## Transitioning status

```bash
# 1. List available transitions for the current state
curl -s -H "Authorization: Bearer $CUTLASS_JIRA_PAT" \
  "$CUTLASS_JIRA_URL/rest/api/2/issue/CUTLASS9-1234/transitions" | jq '.transitions | map({id, name, to: .to.name})'

# 2. Apply one (use the id from above)
curl -s -X POST \
  -H "Authorization: Bearer $CUTLASS_JIRA_PAT" \
  -H "Content-Type: application/json" \
  "$CUTLASS_JIRA_URL/rest/api/2/issue/CUTLASS9-1234/transitions" \
  -d '{"transition": {"id": "31"}}'
```

## Reading tickets

```bash
# Full issue
curl -s -H "Authorization: Bearer $CUTLASS_JIRA_PAT" \
  "$CUTLASS_JIRA_URL/rest/api/2/issue/CUTLASS9-1234"

# Specific fields only (faster, smaller)
curl -s -H "Authorization: Bearer $CUTLASS_JIRA_PAT" \
  "$CUTLASS_JIRA_URL/rest/api/2/issue/CUTLASS9-1234?fields=summary,status,assignee,priority,components,fixVersions"

# Custom field name → id mapping for an issue
curl -s -H "Authorization: Bearer $CUTLASS_JIRA_PAT" \
  "$CUTLASS_JIRA_URL/rest/api/2/issue/CUTLASS9-1234?expand=names" | jq '.names'

# Comments (paginated)
curl -s -H "Authorization: Bearer $CUTLASS_JIRA_PAT" \
  "$CUTLASS_JIRA_URL/rest/api/2/issue/CUTLASS9-1234/comment?startAt=0&maxResults=50"
```

### JQL search

```bash
# URL-encode the JQL via --data-urlencode
curl -s -G -H "Authorization: Bearer $CUTLASS_JIRA_PAT" \
  --data-urlencode 'jql=project = CUTLASS9 AND issuetype = Bug AND status != Closed AND text ~ "CRI gemm" ORDER BY created DESC' \
  --data-urlencode 'fields=summary,status,priority,assignee' \
  --data-urlencode 'maxResults=20' \
  "$CUTLASS_JIRA_URL/rest/api/2/search" \
  | jq '{total, issues: [.issues[] | {key, summary: .fields.summary, status: .fields.status.name, priority: .fields.priority.name, assignee: (.fields.assignee.displayName // "Unassigned")}]}'
```

**Pagination**: default `maxResults` is 50 (max 100). Use `startAt` and check the `total` field to know if more pages exist.

## Looking up users

```bash
# Search by username, display name, or email
curl -s -H "Authorization: Bearer $CUTLASS_JIRA_PAT" \
  "$CUTLASS_JIRA_URL/rest/api/2/user/search?username=<query>" | jq '.[] | {name, displayName, emailAddress, key}'

# Current user
curl -s -H "Authorization: Bearer $CUTLASS_JIRA_PAT" \
  "$CUTLASS_JIRA_URL/rest/api/2/myself"
```

To assign a ticket: `"assignee": {"name": "<username>"}` (use the `name` from the user search, not the display name).

## Jira wiki markup reference

| Markup | Rendered |
|--------|----------|
| `h1.`, `h2.`, `h3.` | Headings |
| `*bold*` | **bold** |
| `_italic_` | _italic_ |
| `{code}...{code}` | Code block |
| `{code:cpp}...{code}` | Code block with syntax highlighting |
| `{{monospace}}` | Inline code |
| `# item` | Numbered list |
| `* item` | Bullet list |
| `[label\|url]` | Hyperlink |
| `\|col1\|col2\|` | Table row |
| `\|\|col1\|\|col2\|\|` | Table header |
| `{noformat}...{noformat}` | Preformatted block |
| `{quote}...{quote}` | Block quote |
| `[~username]` | User mention |
| `!image.png!` | Inline attached image |

## Error handling

| HTTP | Cause | Fix |
|------|-------|-----|
| 401 | PAT expired/revoked | Regenerate PAT in Jira profile, update `~/.cred` |
| 403 | No permission for project/action | Confirm group membership; may need to be added to CUTLASS9 |
| 400 | Missing required field or invalid value | Inspect `errors` and `errorMessages` in the response body |
| 404 on `/issue/KEY` | Wrong key, or no read permission | Verify the key; try via the web UI |
| 404 on `/createmeta` | Endpoint is restricted on this Jira | Discover fields by reading an existing CUTLASS9 ticket with `?expand=names` |
| "Field not on screen" | Custom field not on the Bug screen | Use a different issue type, or ask a project admin to add the field |

## Quick reference

| Action | Endpoint | Method |
|--------|----------|--------|
| Create issue | `/rest/api/2/issue` | POST |
| Read issue | `/rest/api/2/issue/{key}` | GET |
| Update fields | `/rest/api/2/issue/{key}` | PUT |
| Add comment | `/rest/api/2/issue/{key}/comment` | POST |
| Get transitions | `/rest/api/2/issue/{key}/transitions` | GET |
| Apply transition | `/rest/api/2/issue/{key}/transitions` | POST |
| Link issues | `/rest/api/2/issueLink` | POST |
| JQL search | `/rest/api/2/search` | GET |
| User search | `/rest/api/2/user/search?username=` | GET |
| Project versions | `/rest/api/2/project/CUTLASS9/versions` | GET |
| Project components | `/rest/api/2/project/CUTLASS9/components` | GET |
| Issue link types | `/rest/api/2/issueLinkType` | GET |
