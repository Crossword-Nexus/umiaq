# Release Process

This document describes how to create a new release of Umiaq and deploy it to GitHub Pages.

## Overview

Umiaq uses **tag-based deployment**. When you push a version tag (e.g., `v0.2.0`), GitHub Actions automatically:
1. Validates the tag matches the version in `Cargo.toml`
2. Builds the WASM binary
3. Deploys to GitHub Pages at https://crossword-nexus.github.io/umiaq/

## Prerequisites

- Write access to the repository
- Clean working directory (i.e., `git status` shows no uncommitted changes)
- All changes merged to `main` branch

## Release Steps

### 1. Update the Version

Edit `Cargo.toml` and increment the version following [Semantic Versioning](https://semver.org/):

```toml
[package]
version = "0.2.0"  # change this
```

**Version Guidelines:**
- **MAJOR** (1.0.0): breaking changes to the pattern syntax or API
- **MINOR** (0.2.0): new features, backward-compatible
- **PATCH** (0.1.1): bug fixes, backward-compatible

### 2. Commit the Version Bump

```bash
git add Cargo.toml
git commit -m "Bump version to 0.2.0"
```

### 3. Create and Push the Tag

```bash
# Create the tag (must match Cargo.toml version with 'v' prefix)
git tag v0.2.0

# Push the tag to trigger deployment
git push origin v0.2.0
```

**Important:** The tag must match the version in `Cargo.toml`:
- `Cargo.toml`: `version = "0.2.0"`
- Git tag: `v0.2.0`

If they don't match, the deployment will fail with a validation error.

### 4. Monitor the Deployment

1. Go to the [Actions tab](https://github.com/Crossword-Nexus/umiaq/actions)
2. Find the "Pages" workflow run for your tag
3. Wait for both the "build" and "deploy" jobs to complete (usually 2-3 minutes)
4. Verify the deployment at https://crossword-nexus.github.io/umiaq/

## Troubleshooting

### Version Mismatch Error

**Error:** e.g., `Tag version (0.2.0) does not match Cargo.toml version (0.1.0)`

**Solution:**
1. Delete the tag: `git tag -d v0.2.0 && git push origin :refs/tags/v0.2.0`
2. Update `Cargo.toml` to the correct version
3. Commit and create the tag again

### Deployment Didn't Trigger

**Issue:** Pushed a tag but no deployment happened

**Solution:** Verify the tag starts with `v`:
```bash
# ✅ correct
git tag v0.2.0

# ❌ wrong (won't trigger deployment)
git tag 0.2.0
```

### Need to Rollback

**Solution:** Push an older tag to redeploy a previous version:
```bash
# Deploy version 0.1.0 again
git push origin v0.1.0 --force
```

## Example Release Session

```bash
# Starting from clean main branch
$ git status
On branch main
nothing to commit, working tree clean

# Update version in Cargo.toml (0.1.0 → 0.2.0)
$ vim Cargo.toml

# Commit the version bump
$ git add Cargo.toml
$ git commit -m "Bump version to 0.2.0"

# Push to main first (optional, but recommended)
$ git push origin main

# Create and push the tag
$ git tag v0.2.0
$ git push origin v0.2.0

# Monitor deployment
$ open https://github.com/Crossword-Nexus/umiaq/actions
```

## Testing Before Release

Before creating a release tag, you can test the build by:
1. Creating a pull request to `main`
2. The Pages workflow will build (but not deploy) the WASM
3. Review the build logs to ensure no errors

## Release Checklist

- [ ] Version updated in `Cargo.toml`
- [ ] Version follows SemVer (MAJOR.MINOR.PATCH)
- [ ] Changes committed to `main`
- [ ] Tag created with `v` prefix matching Cargo.toml version
- [ ] Tag pushed to GitHub
- [ ] GitHub Actions workflow completed successfully
- [ ] Deployment verified at https://crossword-nexus.github.io/umiaq/
- [ ] Error reports show correct version and build timestamp
