# AGENTS.md

> **Source of truth: [CLAUDE.md](CLAUDE.md).** Read it first. This file is a short pointer
> for any coding agent; the project map, conventions, and gotchas all live in CLAUDE.md.

## TL;DR for agents

- **What this is:** an educational, pure-NumPy comparison of three discretizations of
  gradient flow (Explicit / Implicit / Newton GD). See [CLAUDE.md](CLAUDE.md) for the full
  layout, architecture notes, and known bugs.
- **Treat the code as a frozen milestone.** Do **not** refactor or "fix" the implementation
  in `custom_library/` unless explicitly asked — the known issues (e.g. reference-not-copy in
  `History.save_model`, `metrics.accuracy()` returning MAE, no random seed) are documented
  and intentionally left in place. Flag them; don't silently change them.
- **Scope of edits:** docs, scaffolding (`requirements.txt`, `.gitignore`, `LICENSE`), and the
  `README.md` report are fair game. The math in the README is the author's — fix clear typos
  and broken LaTeX, but preserve the derivations and voice.
- **Running it:** `pip install -r requirements.txt`, then open `main.ipynb` from the repo
  root (it appends the project root to `sys.path` and imports `custom_library`).

For anything not covered here, defer to [CLAUDE.md](CLAUDE.md).
