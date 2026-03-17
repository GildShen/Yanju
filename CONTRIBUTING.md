# Contributing

This repository can be maintained directly on `main`, but the default recommended workflow is:

1. Create a short-lived branch from `main`
2. Make one focused change
3. Open a pull request into `main`
4. Merge after CI passes

## Branch Naming

Use descriptive branch names with a simple prefix:

- `feat/<topic>`
- `fix/<topic>`
- `docs/<topic>`
- `chore/<topic>`

Examples:

- `feat/web-search-filters`
- `fix/readme-encoding`
- `docs/setup-notes`

## Pull Request Scope

Keep each PR small enough to review quickly.

Good PRs usually:

- change one feature or one bug
- include a short summary of what changed
- mention any manual verification steps
- avoid unrelated formatting-only edits

## Local Checks

Before opening a PR, run:

```bash
python -m compileall research_agent
python -m research_agent.cli --help
```

If your change affects the web UI, also run:

```bash
python -m research_agent.web
```

Then verify the local UI still loads at `http://127.0.0.1:8000`.

## Pull Request Process

Typical flow:

```bash
git checkout main
git pull --ff-only origin main
git checkout -b fix/my-change
# edit files
git add .
git commit -m "fix: describe the change"
git push -u origin fix/my-change
```

Then open a pull request from your branch into `main`.

## Merge Strategy

Prefer squash merge for small, focused PRs so `main` stays readable.
