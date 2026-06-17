---
name: cutlass-confluence-wiki
description: Use when reading, searching, creating, updating, commenting on, or attaching files to Confluence pages on Intel's internal wiki (wiki.ith.intel.com). Default space is CUTLASS — the SYCL*TLA Wiki. The PAT is workstation-local and authorizes every Confluence space the user can access on that host, so the same skill applies to other spaces by pointing at a different parent page or space key.
---

# CUTLASS Confluence Wiki

Read and write Confluence pages on Intel's internal wiki at `https://wiki.ith.intel.com`. Primary entry point is the **SYCL\*TLA Wiki** (the "CUTLASS" space). Companion to [`cutlass-jira/SKILL.md`](../cutlass-jira/SKILL.md) — design / RCA / how-to content goes on the wiki, tracking goes on Jira, and the two should cross-link.

## Credentials and safety

Each contributor stores their own Personal Access Token (PAT) on their workstation in `~/.cred` (mode `600`):

```
CONFLUENCE_PAT=<your-personal-access-token>
CUTLASS_CONFLUENCE_URL=https://wiki.ith.intel.com
```

`~/.cred` is **outside the repository** and never committed.

The PAT is a single token that authorizes every Confluence space the user can access on `wiki.ith.intel.com` — not specific to the CUTLASS space. To target a different space on the same host, reuse `CONFLUENCE_PAT` and override the base URL inline (or add a new `<PROJECT>_CONFLUENCE_URL` if the URL differs).

**Hard rules:**
- **Never embed a PAT in this skill, in any file in the repository, in commit messages, or in PR descriptions.** The skill always references the token as `$CONFLUENCE_PAT` — a shell variable populated at runtime.
- **Never write a PAT into a config that gets checked in.**
- If a PAT ever appears in git history (a misclick, a paste into a tracked file, …): rotate it immediately at `$CUTLASS_CONFLUENCE_URL/plugins/personalaccesstokens/usertokens.action`, then scrub history with `git filter-repo` and force-push.

**Always load credentials first** in your shell session (or before the first `curl` command that uses `$CONFLUENCE_PAT` / `$CUTLASS_CONFLUENCE_URL`):

    set -a; source ~/.cred; set +a

### TLS verification

The default in every example below is `curl -s` — silent, with normal TLS verification. Disabling TLS verification (`-k` / `--insecure`) makes man-in-the-middle attacks undetectable, so do not use `-k` casually.

If `curl -s` against `$CUTLASS_CONFLUENCE_URL` fails with a TLS error on your machine, the cause is almost always a missing Intel internal CA chain. The right fix is to install the Intel CA bundle (your IT distribution channel) so verification succeeds. Only as a last resort — when you are certain the connection cannot be intercepted (e.g. on Intel's internal network) and you cannot install the CA — fall back to `curl -sk` (or `--insecure`). Document the fallback in the calling script and treat it as a temporary workaround, not a default.

If a call returns `401 Unauthorized`, the PAT has expired or been revoked. Regenerate at `$CUTLASS_CONFLUENCE_URL/plugins/personalaccesstokens/usertokens.action`, then update `~/.cred`.

## Wiki basics

| Field | Value |
|-------|-------|
| Base URL | `$CUTLASS_CONFLUENCE_URL` (= `https://wiki.ith.intel.com`) |
| API root | `$CUTLASS_CONFLUENCE_URL/rest/api` (Confluence Server REST API v1) |
| Default space key | `CUTLASS` |
| Browse URL pattern | `$CUTLASS_CONFLUENCE_URL/pages/viewpage.action?pageId=<id>` or `$CUTLASS_CONFLUENCE_URL/spaces/<KEY>/pages/<id>/<slug>` |

This is **Confluence Server / Data Center**, not Cloud. The REST API path is `/rest/api/...` and the body format is **storage** (XHTML), not Cloud's ADF.

The SYCL\*TLA Wiki root page id is discoverable rather than hard-coded — title and id can drift, so look it up live:

```bash
curl -s -G -H "Authorization: Bearer $CONFLUENCE_PAT" \
  --data-urlencode "spaceKey=CUTLASS" \
  --data-urlencode "title=SYCL*TLA Wiki" \
  "$CUTLASS_CONFLUENCE_URL/rest/api/content" \
  | jq '.results[] | {id, title}'
```

(The page exists and the title uses `*`, not the URL slug's `+`.)

## Workflow rules

1. **User approval required for any write.** ALWAYS print the full JSON payload (including the rendered storage-format body if it's long) and ask for explicit confirmation before:
   - creating a page
   - updating a page
   - adding/editing/deleting a comment
   - uploading an attachment
   - moving/deleting a page

   Never write to Confluence silently.

2. **AI-generated disclaimer.** Every page body and comment created via this skill MUST end with one of:
   - `<p><em>This page was created with AI assistance.</em></p>`
   - `<p><em>This page section was edited with AI assistance.</em></p>`
   - `<p><em>This comment was posted with AI assistance.</em></p>`

3. **Version bump on update.** Confluence requires the new version number to be the *current version + 1*. Always read the current page first to get `version.number`, then send `version.number + 1`. A wrong version returns `409 Conflict` — refetch and retry, do not just bump blindly.

4. **Verify the target before writing.** Before creating a child page, fetch the parent's `id`, `space.key`, and `title` to confirm it's the page you think it is. Before updating, fetch the existing page and diff your intended body against it — never overwrite without seeing what's there.

5. **Confidentiality check.** Pages with markers like `[Intel Top Secret]`, `[Intel Confidential]`, or restricted view permissions exist in this space. Before reading-then-summarizing one for an external audience (PR description, public commit message, third-party LLM), confirm with the user that the summary is appropriate to share.

6. **Report results.** After creating/updating, return the page id, title, browse URL (`$CUTLASS_CONFLUENCE_URL/pages/viewpage.action?pageId=<id>`), and new version number.

## Reading pages

### By id (most reliable)

```bash
set -a; source ~/.cred; set +a

curl -s -H "Authorization: Bearer $CONFLUENCE_PAT" \
  "$CUTLASS_CONFLUENCE_URL/rest/api/content/<PAGE_ID>?expand=body.storage,version,space,ancestors" \
  | jq '{id, title, space: .space.key, version: .version.number, ancestors: [.ancestors[] | {id, title}], body_len: (.body.storage.value | length)}'
```

`expand` options worth knowing:
- `body.storage` — XHTML storage format (what to send back on update)
- `body.view` — rendered HTML (read-only, do not send back)
- `body.export_view` — HTML suitable for export
- `version` — current version number, required for updates
- `space` — space key/name
- `ancestors` — parent chain
- `children.page` — direct child pages
- `metadata.labels` — labels on the page

### By title within a space

```bash
curl -s -G -H "Authorization: Bearer $CONFLUENCE_PAT" \
  --data-urlencode "spaceKey=CUTLASS" \
  --data-urlencode "title=<exact-title>" \
  --data-urlencode "expand=version,body.storage" \
  "$CUTLASS_CONFLUENCE_URL/rest/api/content" \
  | jq '.results[] | {id, title, version: .version.number}'
```

### Children of a page (one level)

```bash
curl -s -H "Authorization: Bearer $CONFLUENCE_PAT" \
  "$CUTLASS_CONFLUENCE_URL/rest/api/content/<PAGE_ID>/child/page?limit=100&expand=version" \
  | jq '.results[] | {id, title, version: .version.number}'
```

Pagination: response includes `_links.next` if there are more results. Follow it (or pass `start=N` and check `size`/`limit`).

### Descendants

The `/descendant/page` endpoint returns **HTTP 500** on this Confluence instance — do not use it. Walk the tree level-by-level via `/child/page` instead. The CUTLASS space tree is shallow (~3 levels), so a small BFS finishes in seconds:

```bash
ROOT=<root-page-id>
mkdir -p /tmp/conf-bfs
> /tmp/conf-bfs/tree.jsonl

curl -s -H "Authorization: Bearer $CONFLUENCE_PAT" \
  "$CUTLASS_CONFLUENCE_URL/rest/api/content/$ROOT/child/page?limit=200" \
  | jq -c --arg pid "$ROOT" '.results[] | {id, title, parent_id: $pid, depth: 1}' \
  >> /tmp/conf-bfs/tree.jsonl

for d in 2 3 4 5 6; do
  prev=$((d - 1)); before=$(wc -l < /tmp/conf-bfs/tree.jsonl)
  while IFS= read -r row; do
    pid=$(echo "$row" | jq -r .id)
    curl -s -H "Authorization: Bearer $CONFLUENCE_PAT" \
      "$CUTLASS_CONFLUENCE_URL/rest/api/content/$pid/child/page?limit=200" \
      | jq -c --arg pid "$pid" --argjson d "$d" \
          '.results[]? | {id, title, parent_id: $pid, depth: $d}' \
      >> /tmp/conf-bfs/tree.jsonl
  done < <(jq -c "select(.depth == $prev)" /tmp/conf-bfs/tree.jsonl)
  [ "$(wc -l < /tmp/conf-bfs/tree.jsonl)" = "$before" ] && break
done
```

### Search via CQL

```bash
curl -s -G -H "Authorization: Bearer $CONFLUENCE_PAT" \
  --data-urlencode 'cql=space = "CUTLASS" AND type = "page" AND text ~ "flash attention" ORDER BY lastmodified DESC' \
  --data-urlencode 'limit=25' \
  "$CUTLASS_CONFLUENCE_URL/rest/api/content/search" \
  | jq '.results[] | {id, title, url: ._links.webui, lastModified}'
```

Note that the search response is *flat* (`.results[].id`, `.results[].title`), not nested under `.content` — that's the v1 Server schema.

CQL clauses worth knowing: `space`, `type` (`page`, `blogpost`, `comment`, `attachment`), `title`, `text`, `label`, `creator`, `contributor`, `lastmodified`, `created`, `parent`, `ancestor`. Combine with `AND`/`OR`. Sort with `ORDER BY lastmodified|created|title`.

`ancestor = "<id>"` is especially useful — it returns everything under a given page, transitively. To search only inside a specific subtree:

```
cql=ancestor = "<root-id>" AND type = "page" AND text ~ "your terms"
```

## Creating a page

The body MUST be **storage format** (XHTML). Confluence rejects loose HTML / wiki markup unless you explicitly request a different `representation`.

```bash
set -a; source ~/.cred; set +a

curl -s -X POST \
  -H "Authorization: Bearer $CONFLUENCE_PAT" \
  -H "Content-Type: application/json" \
  "$CUTLASS_CONFLUENCE_URL/rest/api/content" \
  -d '{
    "type": "page",
    "title": "[CRI] FlashAttention v0.9.2-cri Perf Notes",
    "space": {"key": "CUTLASS"},
    "ancestors": [{"id": "<parent-id>"}],
    "body": {
      "storage": {
        "value": "<h2>Summary</h2><p>Notes on FA prefill perf for CRI/Xe3p as of v0.9.2-cri.</p><p><em>This page was created with AI assistance.</em></p>",
        "representation": "storage"
      }
    }
  }'
```

Response includes `id`, `title`, `_links.webui`, and `version.number` (=1). Build the browse URL as `$CUTLASS_CONFLUENCE_URL/pages/viewpage.action?pageId=<id>`.

### Title rules

- Titles must be **unique within a space** — duplicates return `400`.
- Use the same architecture prefixes as in [`cutlass-jira/SKILL.md`](../cutlass-jira/SKILL.md): `[CRI]` (Xe3p), `[JGS]`, `[BMG]` (Xe2 / Xe20), `[PVC]` (Xe-HPC / Xe12), `[Xe4]` for the future Xe4 architecture, no prefix for mainline / cross-arch. This makes wiki pages grep-compatible with Jira summaries.

### Parent selection

`ancestors: [{"id": "<parentId>"}]` puts the new page directly under that parent. The full ancestor chain is computed by Confluence — you only need to specify the immediate parent. Look up the appropriate parent for your content area before posting:

```bash
# Top-level sections of the SYCL*TLA Wiki (refresh as the wiki evolves)
SYCL_TLA_ROOT=$(curl -s -G -H "Authorization: Bearer $CONFLUENCE_PAT" \
  --data-urlencode "spaceKey=CUTLASS" --data-urlencode "title=SYCL*TLA Wiki" \
  "$CUTLASS_CONFLUENCE_URL/rest/api/content" | jq -r '.results[0].id')

curl -s -H "Authorization: Bearer $CONFLUENCE_PAT" \
  "$CUTLASS_CONFLUENCE_URL/rest/api/content/$SYCL_TLA_ROOT/child/page?limit=100" \
  | jq '.results[] | {id, title}'
```

Typical placement at time of writing:

| New page is about… | Default parent (look up by title) |
|--------------------|-----------------------------------|
| CRI / Xe3p kernel work, perf, RCAs | *CRI Kernel Enablements* |
| Xe4 design, JGS work, BMG/Xe2 layout/copy-atom | *Xe4/JGS Enablements* |
| New env / build setup how-to | *Quick Start* |
| Dev VM, NFS, server runbook, OOM workarounds | *For Developers* |
| Onboarding-only material | *New Developer Onboarding Guide* |
| Single-question FAQ entry | *Frequently Asked Questions (FAQ)* |
| Superseded content being archived | *Archive* |
| Truly cross-cutting / unsure | Ask the user — wrong parent makes pages hard to find |

When in doubt, search the wiki first — there's usually an existing parent that fits.

## Updating a page

You must send: `id`, `type`, `title`, `space`, `version.number = current + 1`, and `body.storage.value`. **Title and body are both required even if only one changes** — Confluence treats updates as a full replacement.

```bash
set -a; source ~/.cred; set +a

# 1. Read current state
curl -s -H "Authorization: Bearer $CONFLUENCE_PAT" \
  "$CUTLASS_CONFLUENCE_URL/rest/api/content/<PAGE_ID>?expand=body.storage,version,space" \
  > /tmp/page.json

CURR_VER=$(jq -r '.version.number' /tmp/page.json)
NEW_VER=$((CURR_VER + 1))
TITLE=$(jq -r '.title' /tmp/page.json)
SPACE=$(jq -r '.space.key' /tmp/page.json)

# 2. Update
curl -s -X PUT \
  -H "Authorization: Bearer $CONFLUENCE_PAT" \
  -H "Content-Type: application/json" \
  "$CUTLASS_CONFLUENCE_URL/rest/api/content/<PAGE_ID>" \
  -d "$(jq -n --arg t "$TITLE" --arg s "$SPACE" --argjson v $NEW_VER \
    '{
      id: "<PAGE_ID>",
      type: "page",
      title: $t,
      space: {key: $s},
      version: {number: $v},
      body: {
        storage: {
          value: "<p>New body XHTML here, ending with the AI disclaimer.</p><p><em>This page section was edited with AI assistance.</em></p>",
          representation: "storage"
        }
      }
    }')"
```

If the response is `409 Conflict — version`, someone else updated the page between read and write. Refetch and retry.

### Minor edits

Add `"version": {"number": NEW_VER, "minorEdit": true}` to suppress watcher email notifications. Use this for typo fixes, formatting cleanup, AI-disclaimer additions.

## Comments

```bash
# Add an inline comment to a page
curl -s -X POST \
  -H "Authorization: Bearer $CONFLUENCE_PAT" \
  -H "Content-Type: application/json" \
  "$CUTLASS_CONFLUENCE_URL/rest/api/content" \
  -d '{
    "type": "comment",
    "container": {"id": "<PAGE_ID>", "type": "page"},
    "body": {
      "storage": {
        "value": "<p>Reproduced on the CRI sim with the cri_benchmark profile. Posting numbers in a child page shortly.</p><p><em>This comment was posted with AI assistance.</em></p>",
        "representation": "storage"
      }
    }
  }'
```

Read comments on a page:

```bash
curl -s -H "Authorization: Bearer $CONFLUENCE_PAT" \
  "$CUTLASS_CONFLUENCE_URL/rest/api/content/<PAGE_ID>/child/comment?expand=body.storage,version,history.createdBy&limit=50" \
  | jq '.results[] | {id, by: .history.createdBy.displayName, when: .history.createdDate, text: .body.storage.value}'
```

## Attachments

```bash
# Upload (note the X-Atlassian-Token header — required to bypass XSRF check)
curl -s -X POST \
  -H "Authorization: Bearer $CONFLUENCE_PAT" \
  -H "X-Atlassian-Token: nocheck" \
  -F "file=@/path/to/local.png" \
  -F "comment=Trace from cri_benchmark run on <date>" \
  "$CUTLASS_CONFLUENCE_URL/rest/api/content/<PAGE_ID>/child/attachment"
```

To replace an existing attachment with a new version, POST to `/rest/api/content/<PAGE_ID>/child/attachment/<ATTACHMENT_ID>/data` with the same multipart form.

List attachments:

```bash
curl -s -H "Authorization: Bearer $CONFLUENCE_PAT" \
  "$CUTLASS_CONFLUENCE_URL/rest/api/content/<PAGE_ID>/child/attachment?limit=50" \
  | jq '.results[] | {id, title, size: .extensions.fileSize, mediaType: .extensions.mediaType, downloadUrl: ._links.download}'
```

Download an attachment (the `_links.download` path is relative to the base URL):

```bash
curl -s -L -H "Authorization: Bearer $CONFLUENCE_PAT" \
  "$CUTLASS_CONFLUENCE_URL<DOWNLOAD_PATH>" \
  -o /tmp/attachment.png
```

## Labels

```bash
# Add labels (idempotent)
curl -s -X POST \
  -H "Authorization: Bearer $CONFLUENCE_PAT" \
  -H "Content-Type: application/json" \
  "$CUTLASS_CONFLUENCE_URL/rest/api/content/<PAGE_ID>/label" \
  -d '[{"prefix": "global", "name": "cri"}, {"prefix": "global", "name": "perf"}]'

# List
curl -s -H "Authorization: Bearer $CONFLUENCE_PAT" \
  "$CUTLASS_CONFLUENCE_URL/rest/api/content/<PAGE_ID>/label" \
  | jq '.results[] | .name'

# Remove
curl -s -X DELETE -H "Authorization: Bearer $CONFLUENCE_PAT" \
  "$CUTLASS_CONFLUENCE_URL/rest/api/content/<PAGE_ID>/label?name=cri"
```

Recommended label conventions (mirror Jira fix-version suffixes — see [`cutlass-jira/SKILL.md`](../cutlass-jira/SKILL.md)):
- `cri`, `bmg`, `pvc`, `jgs`, `xe4`, `mainline` — architecture
- `perf`, `correctness`, `numerics`, `tooling`, `infra` — area
- `meeting-notes`, `design`, `runbook`, `onboarding` — page kind

## Deleting / moving pages

**Always confirm with the user first** — pages have history but moves and deletes still surprise watchers.

```bash
# Trash a page (recoverable from space admin → Trash for ~30 days, depending on instance config)
curl -s -X DELETE -H "Authorization: Bearer $CONFLUENCE_PAT" \
  "$CUTLASS_CONFLUENCE_URL/rest/api/content/<PAGE_ID>"

# Move (re-parent) — same as update with a new ancestors[]
curl -s -X PUT \
  -H "Authorization: Bearer $CONFLUENCE_PAT" \
  -H "Content-Type: application/json" \
  "$CUTLASS_CONFLUENCE_URL/rest/api/content/<PAGE_ID>" \
  -d '{
    "id": "<PAGE_ID>",
    "type": "page",
    "title": "<existing title>",
    "space": {"key": "CUTLASS"},
    "ancestors": [{"id": "<NEW_PARENT_ID>"}],
    "version": {"number": <CURR + 1>},
    "body": {"storage": {"value": "<existing body>", "representation": "storage"}}
  }'
```

## Confluence storage format — quick reference

The body is XHTML with a few Confluence-specific macros. Common patterns:

| Want | Storage format |
|------|----------------|
| Heading | `<h1>...</h1>` … `<h6>...</h6>` |
| Bold / italic | `<strong>...</strong>` / `<em>...</em>` |
| Inline code | `<code>...</code>` |
| Numbered list | `<ol><li>...</li></ol>` |
| Bullet list | `<ul><li>...</li></ul>` |
| Hyperlink | `<a href="https://...">label</a>` |
| Link to another Confluence page | `<ac:link><ri:page ri:content-title="Page Title" ri:space-key="CUTLASS"/></ac:link>` |
| Link to Jira issue | `<ac:link><ri:url ri:value="https://jira.devtools.intel.com/browse/CUTLASS9-1234"/><ac:plain-text-link-body><![CDATA[CUTLASS9-1234]]></ac:plain-text-link-body></ac:link>` |
| Code block (syntax highlighted) | `<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">cpp</ac:parameter><ac:plain-text-body><![CDATA[ ...code... ]]></ac:plain-text-body></ac:structured-macro>` |
| Info / Note / Warning panel | `<ac:structured-macro ac:name="info"><ac:rich-text-body><p>...</p></ac:rich-text-body></ac:structured-macro>` (also `note`, `warning`, `tip`) |
| Table | `<table><tbody><tr><th>H</th></tr><tr><td>cell</td></tr></tbody></table>` |
| Table of contents | `<ac:structured-macro ac:name="toc"/>` |
| Inline image (attachment) | `<ac:image><ri:attachment ri:filename="file.png"/></ac:image>` |
| Status lozenge | `<ac:structured-macro ac:name="status"><ac:parameter ac:name="colour">Green</ac:parameter><ac:parameter ac:name="title">Done</ac:parameter></ac:structured-macro>` |

Always wrap free-form text in `<p>...</p>`. Loose text outside a block tag is invalid storage and will be rejected.

When authoring long bodies, write them to a temp file and inject via `jq` to avoid quoting hell:

```bash
cat > /tmp/body.xhtml <<'EOF'
<h2>Summary</h2>
<p>Body goes here.</p>
<p><em>This page was created with AI assistance.</em></p>
EOF

jq -n --rawfile body /tmp/body.xhtml --arg parent "<parent-id>" \
  '{type: "page", title: "...", space: {key: "CUTLASS"}, ancestors: [{id: $parent}],
    body: {storage: {value: $body, representation: "storage"}}}' \
  > /tmp/payload.json

curl -s -X POST -H "Authorization: Bearer $CONFLUENCE_PAT" -H "Content-Type: application/json" \
  --data @/tmp/payload.json "$CUTLASS_CONFLUENCE_URL/rest/api/content"
```

## Linking back to Jira

Pages that document a bug, RCA, or design tied to a CUTLASS9 ticket should:

1. Include the Jira key in the title (e.g. `[CRI] FA tile epilogue mismatch — CUTLASS9-1234 RCA`).
2. Link to the ticket in the first paragraph using the Jira issue link macro:
   ```xml
   <p>Tracking ticket: <ac:link><ri:url ri:value="https://jira.devtools.intel.com/browse/CUTLASS9-1234"/><ac:plain-text-link-body><![CDATA[CUTLASS9-1234]]></ac:plain-text-link-body></ac:link></p>
   ```
3. Optionally add a comment back on the Jira ticket pointing to the new wiki page (use [`cutlass-jira/SKILL.md`](../cutlass-jira/SKILL.md)).

## Error handling

| HTTP | Cause | Fix |
|------|-------|-----|
| 401 | PAT expired/revoked | Regenerate at `$CUTLASS_CONFLUENCE_URL/plugins/personalaccesstokens/usertokens.action`, update `~/.cred` |
| 403 | No write permission for the space/page | Confirm space membership; ask space admin |
| 400 — duplicate title | Title already exists in the space | Pick a different title or update the existing page |
| 400 — invalid storage | Body XHTML is malformed (e.g. unclosed tag, raw text outside `<p>`) | Validate the body — every block must be balanced; wrap loose text in `<p>` |
| 404 on `/content/<id>` | Wrong id, page deleted, or no read permission | Verify via web UI; the id in the URL is authoritative |
| 409 — version | Page changed between your read and write | Refetch, increment from the new `version.number`, retry |
| 415 on attachment upload | Missing `X-Atlassian-Token: nocheck` header | Add the header — Confluence's XSRF check requires it |
| 500 on `/descendant/page` | Endpoint is broken on this instance | Walk the tree level-by-level via `/child/page` (see "Descendants" above) |

## Quick reference

| Action | Endpoint | Method |
|--------|----------|--------|
| Get page | `/rest/api/content/{id}` | GET |
| Find page by title | `/rest/api/content?spaceKey=&title=` | GET |
| List children | `/rest/api/content/{id}/child/page` | GET |
| CQL search | `/rest/api/content/search?cql=...` | GET |
| Create page | `/rest/api/content` | POST |
| Update page | `/rest/api/content/{id}` | PUT |
| Trash page | `/rest/api/content/{id}` | DELETE |
| Add comment | `/rest/api/content` (type=comment) | POST |
| List comments | `/rest/api/content/{id}/child/comment` | GET |
| Upload attachment | `/rest/api/content/{id}/child/attachment` | POST (multipart) |
| List attachments | `/rest/api/content/{id}/child/attachment` | GET |
| Add labels | `/rest/api/content/{id}/label` | POST |
| Remove label | `/rest/api/content/{id}/label?name=` | DELETE |
| Current user | `/rest/api/user/current` | GET |
