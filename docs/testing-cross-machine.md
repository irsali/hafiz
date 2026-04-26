# Cross-machine + cross-agent test (Windows + Cursor)

This is a one-page checklist for installing Hafiz on a fresh machine and
driving it from an agent other than Claude Code. It doubles as a test plan:
**capture friction as you hit it**, so the artifact at the end is real
findings, not vibes.

The path below is written for **Windows + Cursor + no GPU**. Adapt the
package-manager line for macOS/Linux as needed.

---

## 0. Define success up-front

Before you start, write down what "test passed" means. Sample bar:

- [ ] `hafiz status --diagnose` is all green.
- [ ] `hafiz ingest .` completes on a non-trivial repo without error.
- [ ] `hafiz agent install cursor --local` writes `.cursor/rules/hafiz.mdc`
      and Cursor loads it (you can see "Hafiz workspace intelligence" listed
      in Cursor's rules panel).
- [ ] In a Cursor chat, asking *"use hafiz to find where authentication is
      handled"* triggers a `hafiz query` / `hafiz context` invocation and
      the model uses the result.
- [ ] At least one `hafiz observe --type decision --source agent:cursor`
      lands during the session.

If any of those fail, that's a test finding — record it (see §6).

---

## 1. Prerequisites (Windows)

- [ ] **Python 3.12+** — install from
      <https://www.python.org/downloads/windows/>. Tick *Add Python to PATH*
      in the installer.
- [ ] **Git** — <https://git-scm.com/download/win>.
- [ ] **Docker Desktop** — <https://docs.docker.com/desktop/install/windows-install/>.
      Required for the easy Postgres + pgvector path. (Native Postgres on
      Windows works but you'll need to build pgvector from source — skip
      unless you really don't want Docker.)
- [ ] **pipx** — in PowerShell:
      ```powershell
      python -m pip install --user pipx
      python -m pipx ensurepath
      # restart PowerShell so PATH refreshes
      ```
- [ ] **Cursor** — <https://www.cursor.com/>. Sign in once so it has a
      working session.

---

## 2. Install Hafiz

In a new PowerShell window:

```powershell
pipx install "git+https://github.com/irsali/hafiz.git"
hafiz --version
```

Skip the `[gpu]` extra — this machine has no GPU, so CPU embeddings via
ONNX is the right path. (If you accidentally install `[gpu]`, the
embedding probe falls back to CPU but the install pulls extra MB you don't
need.)

---

## 3. Start Postgres

```powershell
docker run -d `
  --name hafiz-db `
  -e POSTGRES_USER=postgres `
  -e POSTGRES_PASSWORD=postgres `
  -e POSTGRES_DB=hafiz `
  -p 5432:5432 `
  --restart unless-stopped `
  pgvector/pgvector:pg17
```

PowerShell uses backticks for line continuation; if you'd rather paste it
all on one line, drop the backticks.

Confirm it's up:

```powershell
docker ps | Select-String hafiz-db
```

---

## 4. Configure Hafiz

Hafiz reads `hafiz.toml` from cwd → `~/.config/hafiz/hafiz.toml` →
`/etc/hafiz/hafiz.toml`. On Windows, `~` is `%USERPROFILE%` (typically
`C:\Users\<you>`), so the global config lives at
`C:\Users\<you>\.config\hafiz\hafiz.toml`.

```powershell
mkdir $env:USERPROFILE\.config\hafiz -Force
notepad $env:USERPROFILE\.config\hafiz\hafiz.toml
```

Paste, then update the `root` and `projects` lines:

```toml
[database]
url = "postgresql+asyncpg://postgres:postgres@localhost:5432/hafiz"

[embedding]
model = "nomic-ai/nomic-embed-text-v1.5"
provider = "fastembed"
dimensions = 768
device = "cpu"

[workspace]
root = "C:/Users/<you>/code"          # forward slashes are fine
projects = ["my-project"]
ignore = [".git", "node_modules", "__pycache__", ".venv", "dist", "build"]
```

Note: `device = "cpu"` is explicit here — the default `"auto"` would also
land on CPU, but locking it skips the GPU probe on every cold start.

---

## 5. Initialise + smoke test

```powershell
hafiz init                      # creates schema + pgvector extension
hafiz status --diagnose         # all checks should pass
hafiz doctor --probe            # shows host capabilities + tunable hints
```

First run will download the embedding model (~440 MB to
`%USERPROFILE%\.cache\fastembed\` or similar). Expect a multi-minute
delay and no progress bar. **If this hangs silently for >5 minutes that's
a test finding — record it.**

---

## 6. Wire up Cursor

```powershell
cd <a project you want to ingest>
hafiz agent install cursor --local
```

This writes `.cursor/rules/hafiz.mdc` (with `alwaysApply: true`
frontmatter). Open the project in Cursor — you should see the rule in the
*Rules* panel (Settings → Rules, or Cmd/Ctrl+Shift+P → "Cursor: View
Rules").

Then ingest:

```powershell
hafiz ingest . --project my-project
```

---

## 7. Drive a real task from Cursor

In a Cursor chat, ask something that *should* make the model reach for
hafiz, e.g.:

- "Use hafiz to find where authentication is handled in this repo."
- "Before refactoring `UserService`, check what depends on it via hafiz."
- "What decisions have I recorded about the auth system?"

Watch what Cursor does:

- Does it invoke `hafiz` at all? Or does it skip the rule?
- Does it use `--json`? (The rule says it should.)
- Does it record an `observe` after deciding something?

---

## 8. Capture findings as you go

Treat every surprise as a finding. Use hafiz itself to log them — you
keep a real trail and you exercise the tool at the same time:

```powershell
hafiz observe "Cursor ignored the alwaysApply rule on first chat — only picked it up after a Cursor reload" `
  --type warning --source user:irshad --tags cross-machine,cursor

hafiz observe "Embedding model first-download has no progress output, hung for ~3 min" `
  --type warning --source user:irshad --tags cross-machine,first-run
```

When the run is done:

```powershell
hafiz journal --since 1d --tags cross-machine
```

…gives you a clean digest of every friction point you logged. Bring that
back to the main repo and we'll triage what's worth fixing for real.

---

## 9. Things we expect to be rough (don't bother logging these)

These are known gaps — logging them adds noise without insight:

- **Non-Python files don't get AST graph data.** `.ts`, `.js`, `.cs`,
  `.go`, etc. fall through to the whole-file parser; they're searchable
  but `hafiz graph deps` won't have edges for them. Tree-sitter parsers
  are queued — see prior conversation.
- **Cursor rules with `alwaysApply: true` push a lot of context.** The
  whole skills.md goes in. A "compact" Cursor variant is worth doing
  later but isn't ready yet.
- **No Windows-specific install troubleshooting in the README.** That's
  what this run is supposed to surface — anything a copy-paster hits
  *is* the deliverable.

Anything *outside* this list is fair game for a finding.
