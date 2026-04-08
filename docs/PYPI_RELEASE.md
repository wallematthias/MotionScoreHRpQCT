# PyPI Release Workflow

This project publishes to PyPI via GitHub Actions using trusted publishing.

## One-time setup

1. Create a PyPI project named `motionscorehrpqct`.
2. In GitHub repo settings, add an environment named `pypi`.
3. In PyPI project settings, add a trusted publisher for this GitHub repo/workflow:
   - Workflow file: `.github/workflows/publish-pypi.yml`
   - Environment: `pypi`

## Release steps

1. Ensure CI passes on `main`.
2. Update version in:
   - `motionscore/__init__.py`
   - `setup.py`
3. Create and push a version tag:

```bash
git tag v2.1.0
git push origin v2.1.0
```

4. GitHub Actions will:
   - validate that tag matches `motionscore.__version__`
   - build wheel + sdist
   - run `twine check`
   - publish to PyPI from the `publish` job

## Notes

- Publishing is triggered by tags matching `v*`.
- `pyproject.toml` defines the build backend (`setuptools`).
- `ci.yml` runs tests and packaging smoke checks before release.
