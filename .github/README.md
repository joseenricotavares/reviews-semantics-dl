# Runbook: setting up this `.github` in a repo

1. Copy this `.github/` folder into the target repo.

2. In `workflows/ci.yml`, update what's specific to the new repo:
   - the `pip install -e ...` package paths in the `test` job
   - the `package-name` / `package-dir` inputs passed to `./.github/actions/version-check` and `./.github/actions/tag-package`
   - the Dockerfile paths and smoke-test request in the `docker` job

3. In `setup/branch-protection.json`, update `required_status_checks.contexts` if the job names in `ci.yml` differ from `version-check`, `test`, `docker`.

4. Create a fine-grained PAT for the repo (or extend an existing one's repository access): **Pull requests: Read and write**, **Contents: Read-only**. This is required because PRs opened with the default `GITHUB_TOKEN` don't trigger other workflows — `ci.yml`'s `pull_request` event would never fire for the auto-opened PR.

5. `cp .github/.secrets.env.example .github/.secrets.env` and fill in `AUTO_PR_TOKEN=<the PAT from step 4>`. This file is gitignored — never commit it.

6. From the repo root, run:
   ```bash
   bash .github/setup/bootstrap.sh
   ```
   It logs you into `gh` if needed (interactive), then pushes `.secrets.env` as repo secrets and applies `branch-protection.json` to `main`. The account you log in as must have admin rights on the repo.

7. Push a `feature/*` branch and confirm in the **Actions** tab that `ci.yml` runs in `pull_request` context on the auto-opened PR (not only after merging to `main`).
