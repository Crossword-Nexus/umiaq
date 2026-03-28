# Release Process

This document describes how to create a new release of Umiaq and deploy it to GitHub Pages.

## Overview

Umiaq uses a **two-stage deployment** to avoid pushing untested builds directly to the public URL:

1. Pushing a version tag automatically builds and deploys to the **staging URL**: https://crossword-nexus.github.io/umiaq/staging/
2. After verifying staging looks correct, you manually trigger the **"Promote to Production"** workflow to deploy to the public URL: https://crossword-nexus.github.io/umiaq/

## Prerequisites

- Write access to the repository
- Clean working directory (i.e., `git status` shows no uncommitted changes)
- All changes merged to `main` branch

## One-Time Repository Setup

GitHub Pages must be configured to serve from the `gh-pages` branch (not "GitHub Actions"):

1. Go to **Settings → Pages**
2. Under "Source", select **Deploy from a branch**
3. Choose branch **`gh-pages`**, folder **`/ (root)`**
4. Save

This only needs to be done once.

## Release Steps

### 1. Update the Changelog

Edit `CHANGELOG.md`:
1. Rename the `[Unreleased]` section heading to the new version and today's date, e.g., `[0.3.0] - 2026-03-18`
2. Add a fresh `## [Unreleased]` section above it (leave it empty)
3. Update the comparison links at the bottom of the file

```markdown
## [Unreleased]

## [0.3.0] - 2026-03-18
...previous unreleased entries...
```

### 2. Update the Version

Edit `Cargo.toml` and increment the version following [Semantic Versioning](https://semver.org/):

```toml
[package]
version = "0.3.0"  # change this
```

**Version Guidelines:**
- **MAJOR** (1.0.0): breaking changes to the pattern syntax or API
- **MINOR** (0.2.0): new features, backward-compatible
- **PATCH** (0.1.1): bug fixes, backward-compatible

### 3. Commit the Version Bump and Changelog

```bash
git add Cargo.toml CHANGELOG.md
git commit -m "Bump version to 0.3.0"
```

### 4. Create and Push the Tag

```bash
# Create the tag (must match Cargo.toml version with 'v' prefix)
git tag v0.3.0

# Push the tag — this triggers the staging deployment
git push origin v0.3.0
```

**Important:** The tag must match the version in `Cargo.toml`:
- `Cargo.toml`: `version = "0.3.0"`
- Git tag: `v0.3.0`

If they don't match, the deployment will fail with a validation error.

### 5. Verify Staging

1. Go to the [Actions tab](https://github.com/Crossword-Nexus/umiaq/actions)
2. Find the "Pages" workflow run for your tag and wait for it to complete (~2-3 minutes)
3. Open https://crossword-nexus.github.io/umiaq/staging/ and verify the release looks correct

### 6. Promote to Production

Once staging is verified:

1. Go to the [Actions tab](https://github.com/Crossword-Nexus/umiaq/actions)
2. Select the **"Promote to Production"** workflow from the left sidebar
3. Click **"Run workflow"**
4. Enter the version number (e.g. `0.3.0`) and click **"Run workflow"**
5. Wait for it to complete (~2-3 minutes)
6. Verify the deployment at https://crossword-nexus.github.io/umiaq/

## Troubleshooting

### Version Mismatch Error

**Error:** e.g., `Tag version (0.3.0) does not match Cargo.toml version (0.2.3)`

**Solution:**
1. Delete the tag: `git tag -d v0.3.0 && git push origin :refs/tags/v0.3.0`
2. Update `Cargo.toml` to the correct version
3. Commit and create the tag again

### Deployment Didn't Trigger

**Issue:** Pushed a tag but no staging deployment happened

**Solution:** Verify the tag starts with `v`:
```bash
# ✅ correct
git tag v0.3.0

# ❌ wrong (won't trigger deployment)
git tag 0.3.0
```

### Need to Rollback

**Solution:** Run the "Promote to Production" workflow with the previous version number to redeploy it to the production URL.

## Example Release Session

```bash
# Starting from clean main branch
$ git status
On branch main
nothing to commit, working tree clean

# Update CHANGELOG.md: rename [Unreleased] to [0.3.0] - YYYY-MM-DD, add new empty [Unreleased]
$ vim CHANGELOG.md

# Update version in Cargo.toml (0.2.3 → 0.3.0)
$ vim Cargo.toml

# Commit both files
$ git add Cargo.toml CHANGELOG.md
$ git commit -m "Bump version to 0.3.0"

# Push to main first (optional, but recommended)
$ git push origin main

# Create and push the tag — triggers staging deployment
$ git tag v0.3.0
$ git push origin v0.3.0

# Wait for Pages workflow to complete, then check staging
$ open https://crossword-nexus.github.io/umiaq/staging/

# Once satisfied, promote via GitHub Actions UI:
# Actions → "Promote to Production" → Run workflow → version: 0.3.0
```

## Testing Before Release

Before creating a release tag, you can test the build by:
1. Creating a pull request to `main`
2. The Pages workflow will build (but not deploy) the WASM
3. Review the build logs to ensure no errors

## Release Checklist

- [ ] `CHANGELOG.md` updated: `[Unreleased]` renamed to new version + date, fresh `[Unreleased]` section added, comparison links updated
- [ ] Version updated in `Cargo.toml`
- [ ] Version follows SemVer (MAJOR.MINOR.PATCH)
- [ ] `Cargo.toml` and `CHANGELOG.md` committed to `main`
- [ ] Tag created with `v` prefix matching Cargo.toml version
- [ ] Tag pushed to GitHub
- [ ] "Pages" workflow completed successfully
- [ ] Staging verified at https://crossword-nexus.github.io/umiaq/staging/
- [ ] "Promote to Production" workflow run and completed
- [ ] Deployment verified at https://crossword-nexus.github.io/umiaq/
- [ ] Error reports show correct version and build timestamp
