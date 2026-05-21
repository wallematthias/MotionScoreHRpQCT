# Self-Service MotionScore Model Access

MotionScore model access should not depend on a paid form service, a usage-gated signup provider, or manual license issuance.

The current model is:

1. Users generate a local registration key with `motionscore license-register`.
2. The command writes `~/.motionscore/MotionScore/license.json`.
3. The command prints a prefilled tracking URL. Users can submit it so the project can track which people/groups are using the weights.
4. Users download public model bundles with `motionscore model-download`.
5. The downloader registers the extracted checkpoints in `model_registry.json` and writes a local usage event to `~/.motionscore/MotionScore/usage_events.tsv`.

This is intentionally a tracking and attribution workflow, not a brittle DRM workflow. It keeps the model weights available even if the optional tracking endpoint changes.

## User Flow

```bash
motionscore license-register \
  --name "Jane Doe" \
  --institution "UCSF" \
  --email "jane@example.edu" \
  --group "Bone Lab" \
  --intended-use "HR-pQCT motion scoring" \
  --open-tracking-url

motionscore model-download --model-id base-v1
motionscore model-list
```

The default tracking URL is a GitHub issue creation URL. It can be replaced without changing the CLI:

```bash
motionscore license-register \
  --name "Jane Doe" \
  --institution "UCSF" \
  --email "jane@example.edu" \
  --tracking-url "https://github.com/<owner>/<repo>/issues/new"
```

## Model Catalog

By default, `model-download` reads:

```text
https://github.com/wallematthias/MotionScoreHRpQCT/releases/latest/download/model_catalog.json
```

The catalog can also be a local JSON file:

```bash
motionscore model-download --catalog /path/to/model_catalog.json --model-id base-v1
```

Example catalog:

```json
{
  "default_model_id": "base-v1",
  "models": [
    {
      "model_id": "base-v1",
      "display_name": "Base MotionScore v1",
      "version": "v1",
      "domain": "motion-score",
      "description": "Base HR-pQCT motion scoring ensemble",
      "url": "https://github.com/wallematthias/MotionScoreHRpQCT/releases/download/models-v1/base-v1.zip",
      "sha256": "<optional sha256>",
      "make_default": true
    }
  ]
}
```

Supported bundle formats:

- `.zip`
- `.tar`
- `.tar.gz`
- `.tgz`
- single `.pt` checkpoint files

Archives may contain checkpoints directly or inside one top-level directory. The registered model directory must contain `DNN_*.pt` files after extraction.

## What To Track

The tracking submission URL includes:

- name
- institution
- group
- email
- intended use
- generated registration key
- creation timestamp

The local usage event TSV records registration and model-download events. It can be shared by users if a study needs offline audit records.

## Legacy Worker

The old Cloudflare Worker/Supabase/R2 flow remains in `license-worker/` for reference. It is no longer required for basic access because the self-service CLI path avoids a persistent backend dependency.
