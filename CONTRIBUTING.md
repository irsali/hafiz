# Contributing to Hafiz

Thanks for your interest in Hafiz. This guide covers how to report issues,
propose changes, and sign off your commits.

## Reporting issues

Open an issue at https://github.com/irsali/hafiz/issues with:

- What you expected to happen
- What actually happened
- The shortest reproduction you can manage (commands, config, logs)
- Your platform, Python version, and Hafiz version (`hafiz status --json`)

Security-sensitive reports: please email the maintainer privately rather than
filing a public issue.

## Proposing changes

1. **Open an issue first** for anything non-trivial so we can align on scope
   before you write code.
2. **Fork and branch.** One logical change per branch.
3. **Match the existing conventions** in [CLAUDE.md](CLAUDE.md) — async
   end-to-end, `core/` vs. `commands/` split, `--json` on user-facing commands,
   stable JSON shapes documented in [COMMANDS.md](COMMANDS.md).
4. **Run the gates locally** before pushing:
   ```bash
   ruff check .
   ruff format .
   pytest
   ```
5. **Open a PR.** Describe the change, link the issue, call out any
   user-visible or Layer 1 contract impact.

## Developer Certificate of Origin (DCO)

Hafiz is licensed under [FSL-1.1-MIT](LICENSE). To keep the license
enforceable, every contribution must be accompanied by a Developer Certificate
of Origin sign-off — you assert that you wrote the code (or have the right to
submit it) and agree to contribute it under the project's license.

**How to sign off:** add `-s` to every commit.

```bash
git commit -s -m "your message"
```

This appends a trailer like:

```
Signed-off-by: Your Name <you@example.com>
```

The name and email must match your real identity and your git config. PRs
without sign-off on every commit will be asked to amend before merge.

**What you're certifying** — the full DCO 1.1 text:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off)
    is maintained indefinitely and may be redistributed consistent
    with this project and the open source license(s) involved.
```

The canonical source is https://developercertificate.org.

## Code of conduct

Be kind, be specific, assume good faith. Harassment, personal attacks, or
discriminatory language have no place here.
