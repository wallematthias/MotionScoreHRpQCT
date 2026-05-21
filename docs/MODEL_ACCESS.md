# MotionScore Model Access

MotionScore model weights are distributed as public GitHub release assets. There is no local license key, manual approval step, tracking form, or backend service in the download path.

## User Flow

```bash
motionscore model-download --model-id base-v1
motionscore model-list
```

By default, `model-download` reads the catalog attached to the latest GitHub release:

```text
https://github.com/wallematthias/MotionScoreHRpQCT/releases/latest/download/model_catalog.json
```

The catalog points to the actual model bundle, for example:

```text
https://github.com/wallematthias/MotionScoreHRpQCT/releases/download/v2.5.4/motionscore-base-v1.tar.gz
```

The downloader verifies `sha256` when the catalog provides one, extracts the checkpoint files, and registers the model in `model_registry.json` under the selected model root.

## Usage Counts

GitHub release asset download counts are the central usage metric. They can be inspected in the release page or with:

```bash
gh release view v2.5.4 --repo wallematthias/MotionScoreHRpQCT --json assets
```

This reports download counts for `model_catalog.json` and `motionscore-base-v1.tar.gz`.

## Updating Models

1. Create a new release tag.
2. Upload the model bundle as a release asset.
3. Upload `model_catalog.json` with the new URL and checksum.
4. Mark the release as latest.

The CLI and Slicer module use `releases/latest/download/model_catalog.json`, so they automatically pick up the newest catalog once the release is marked latest.
