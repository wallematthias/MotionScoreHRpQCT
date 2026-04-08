# MotionScore License Worker

Cloudflare Worker backend for:

- in-app signup (no email verification mode),
- automatic license grant at signup in current no-email-verification mode,
- license activation with per-device limits,
- encrypted model manifest/file delivery from R2,
- usage event logging to Supabase for usage-tracked model licensing.

## Endpoints

- `GET /health`
- `POST /signup`
- `POST /activate`
- `GET /model/:version/manifest`
- `GET /model/:version`
- `POST /event`

## Setup

1. Copy config and edit:

```bash
cp wrangler.toml.example wrangler.toml
```

2. Install:

```bash
npm install
```

3. Add secrets:

```bash
npx wrangler secret put SUPABASE_SERVICE_ROLE_KEY
npx wrangler secret put JWT_SECRET
npx wrangler secret put MODEL_MASTER_KEY
```

4. Dev server:

```bash
npm run dev
```

5. Deploy:

```bash
npm run deploy
```

Full walkthrough: [`docs/LICENSING_NO_DOMAIN_SETUP.md`](../docs/LICENSING_NO_DOMAIN_SETUP.md)
