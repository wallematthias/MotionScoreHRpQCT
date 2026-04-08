# MotionScore No-Domain Licensing Setup (Cloudflare + Supabase + R2)

This guide gets you running without owning a mail domain yet.

For now, we skip email verification and issue a key immediately on signup.
Later, you can add Resend + verified domain without changing the core API shape.

## 1) Prerequisites

- Cloudflare account with Workers + R2 enabled
- Supabase project created
- Node.js 20+ locally

## 2) Create Supabase schema

1. Open your Supabase project SQL editor.
2. Run:
   - [`docs/LICENSING_SUPABASE_SCHEMA.sql`](./LICENSING_SUPABASE_SCHEMA.sql)

## 3) Create Worker project (already scaffolded here)

Worker source is in:
- [`license-worker/`](../license-worker/)

Install dependencies:

```bash
cd MotionScoreHRpQCT/license-worker
npm install
```

## 4) Configure Cloudflare + env

1. Copy example config:

```bash
cp wrangler.toml.example wrangler.toml
```

2. Edit `wrangler.toml`:
- `account_id`
- `bucket_name` (you said `motionscore-models`)
- `SUPABASE_URL`

3. Set secrets:

```bash
npx wrangler secret put SUPABASE_SERVICE_ROLE_KEY
npx wrangler secret put JWT_SECRET
```

## 5) Upload encrypted model files to R2

Expected object layout:

```text
models/<version>.enc
models/<version>.manifest.json
```

Create encrypted bundle + manifest:

```bash
cd MotionScoreHRpQCT
python3 scripts/encrypt_model_bundle.py --version v1 --input-dir models --output-dir /tmp/motionscore_bundle
```

The script prints a base64 AES key. Set it as Worker secret:

```bash
cd MotionScoreHRpQCT/license-worker
npx wrangler secret put MODEL_MASTER_KEY
```

Example:

```bash
npx wrangler r2 object put motionscore-models/models/v1.enc --file /tmp/motionscore_bundle/v1.enc
npx wrangler r2 object put motionscore-models/models/v1.manifest.json --file /tmp/motionscore_bundle/v1.manifest.json
```

## 6) Deploy Worker

```bash
npx wrangler deploy
```

After deploy, you get base URL like:
- `https://motionscore-license-api.<subdomain>.workers.dev`

## 7) Smoke tests

Signup:

```bash
curl -sS -X POST "$BASE/signup" \
  -H "content-type: application/json" \
  -d '{"name":"Jane Doe","institution":"UCSF","email":"jane@example.com"}'
```

Activate:

```bash
curl -sS -X POST "$BASE/activate" \
  -H "content-type: application/json" \
  -d '{"email":"jane@example.com","license_key":"<FROM_SIGNUP>","device_hash":"mac-abc123"}'
```

Download manifest:

```bash
curl -sS "$BASE/model/v1/manifest" -H "authorization: Bearer <token>"
```

Log prediction:

```bash
curl -sS -X POST "$BASE/event" \
  -H "content-type: application/json" \
  -H "authorization: Bearer <token>" \
  -d '{"event_type":"prediction","payload":{"scan_id":"sub-001_site-tibia_ses-T1_abc","duration_ms":1200}}'
```

## 8) Current policy

- max devices per license: `2`
- duration: `365` days (1 year)
- tracked events: signup, activation, download, prediction

## 9) Upgrade path when you buy a domain

- Add Resend + verified sender domain
- Change `/signup` to issue verification code by email
- Add `/verify` endpoint and require verification before license issuance
