-- MotionScore licensing backend schema (no-domain MVP)
-- Run in Supabase SQL editor.

create extension if not exists pgcrypto;

create table if not exists public.users (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  institution text not null,
  email text not null unique,
  email_verified_at timestamptz,
  created_at timestamptz not null default now()
);

create table if not exists public.licenses (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.users(id) on delete cascade,
  license_key text not null unique,
  status text not null default 'active' check (status in ('active', 'revoked', 'expired')),
  max_devices int not null default 2 check (max_devices > 0),
  expires_at timestamptz not null,
  created_at timestamptz not null default now()
);

create index if not exists idx_licenses_user_status on public.licenses(user_id, status);
create index if not exists idx_licenses_expires_at on public.licenses(expires_at);

create table if not exists public.devices (
  id uuid primary key default gen_random_uuid(),
  license_id uuid not null references public.licenses(id) on delete cascade,
  device_hash text not null,
  last_seen_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  unique (license_id, device_hash)
);

create index if not exists idx_devices_license_id on public.devices(license_id);

create table if not exists public.events (
  id uuid primary key default gen_random_uuid(),
  license_id uuid references public.licenses(id) on delete set null,
  event_type text not null check (event_type in ('signup', 'activation', 'download', 'prediction')),
  payload_json jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_events_license_type on public.events(license_id, event_type);
create index if not exists idx_events_created_at on public.events(created_at);

comment on table public.users is 'License user registry for MotionScore.';
comment on table public.licenses is 'Issued license keys and policy state.';
comment on table public.devices is 'Activated devices per license key.';
comment on table public.events is 'Usage telemetry: signup, activation, download, prediction.';
