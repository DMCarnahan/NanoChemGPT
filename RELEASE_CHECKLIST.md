# Release Checklist

Use this checklist to make reproducible, publication-ready releases.

Pre-release (on a feature branch)
- [ ] Ensure all tests pass locally and in CI (run `pytest -q`).
- [ ] Update `CHANGELOG.md` with a concise summary of changes, PR numbers, and authors.
- [ ] Bump version in the code/package (follow semantic versioning).
- [ ] Ensure `requirements.txt` and `requirements-dev.txt` reflect pinned versions for reproducibility.
- [ ] Run formatting and linting: `ruff check .`, `isort . --profile black`, `black .`.
- [ ] If adding heavy dependencies (faiss, torch), add notes about optional installs and expected platforms.

Release steps (on `main`)
- [ ] Merge PR into `main` and confirm CI completed successfully.
- [ ] Create a git tag (annotated) with the release version:

  git tag -a vX.Y.Z -m "Release vX.Y.Z"
  git push origin vX.Y.Z

- [ ] Create a GitHub Release (draft) from the tag and paste the `CHANGELOG.md` entry.
- [ ] Build Python artifacts and test in a clean environment:

  python -m pip install --upgrade pip
  python -m pip install build
  python -m build --sdist --wheel
  python -m pip install --upgrade twine
  python -m pip install dist/*.whl --no-deps

- [ ] Publish to TestPyPI first (optional) to verify packaging:

  python -m twine upload --repository testpypi dist/*

- [ ] Publish to PyPI (when ready):

  python -m twine upload dist/*

Post-release
- [ ] Create / update the release notes on GitHub, attach artifacts if needed.
- [ ] Update README badges if necessary.
- [ ] Announce release in project channels and record DOI/Zenodo link if publishing.
- [ ] Optionally build and publish Docker image (see Dockerfile template in `scaffold/`).

Secrets and automation
- Add `PYPI_API_TOKEN` as a repository secret to allow `release.yml` to publish to PyPI/TesPyPI via Twine.
- Add `GITHUB_TOKEN` (provided by GitHub Actions automatically) for basic registry writes; for GHCR write access create a `GHCR_PAT` or allow `GITHUB_TOKEN` to write packages.
- Optional: add `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository secrets to enable pushing images to Docker Hub.

Docker publish notes
- The repository includes a `Dockerfile` and a `docker-image.yml` workflow that will build and push to `ghcr.io/${{ github.repository }}` when a tag is pushed. To enable Docker Hub publishing, add `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets.

Automated release notes
- Add the following repository secrets:
  - `PYPI_API_TOKEN` — a PyPI API token with upload permissions.
  - (optional) `PUBLISH_REPOSITORY_URL` — used when publishing to TestPyPI or a custom repository URL.
  - (optional) `GHCR_PAT` or rely on `GITHUB_TOKEN` for GHCR pushes.
  - (optional) `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` — to enable Docker Hub pushes from the `docker-image.yml` workflow.

Quick verification after tagging
1. Create an annotated tag and push: `git tag -a vX.Y.Z -m "Release vX.Y.Z" && git push origin vX.Y.Z`.
2. Monitor `Actions` → `Release (PyPI)` and `Build and publish Docker image` workflows for success.
3. Download the `dist-artifacts-<tag>` artifact from the release workflow to inspect artifacts before PyPI publish.


Notes on reproducibility
- Pin major compiled deps to known-compatible versions (for example `numpy<2` to avoid ABI breakage when building faiss wheels).
- Consider providing a `poetry.lock` or `pip-tools` `requirements.txt` to pin transitive deps.

Troubleshooting
- If CI flakiness occurs in `miner-integration`, re-run the job and capture logs. Consider adding retries or a cached prebuilt spaCy model for CI.
- For faiss installation failures on linux runners, use prebuilt wheels when available or skip faiss in CI and run `miner-integration` on a matrix runner with CPU-only wheel caching.

*** End Patch