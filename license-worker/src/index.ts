export interface Env {
  MODELS_BUCKET: R2Bucket;
  SUPABASE_URL: string;
  SUPABASE_SERVICE_ROLE_KEY: string;
  JWT_SECRET: string;
  MODEL_MASTER_KEY: string;
  MAX_DEVICES?: string;
  LICENSE_DURATION_DAYS?: string;
  MODEL_OBJECT_PREFIX?: string;
  ALLOWED_ORIGIN?: string;
}

type JsonMap = Record<string, unknown>;

type UserRow = {
  id: string;
  name: string;
  institution: string;
  email: string;
  email_verified_at: string | null;
  created_at: string;
};

type LicenseRow = {
  id: string;
  user_id: string;
  license_key: string;
  status: "active" | "revoked" | "expired";
  max_devices: number;
  expires_at: string;
  created_at: string;
};

const DEFAULT_MAX_DEVICES = 2;
const DEFAULT_LICENSE_DURATION_DAYS = 365;

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === "OPTIONS") {
      return withCors(new Response(null, { status: 204 }), env);
    }

    try {
      const url = new URL(request.url);
      const path = url.pathname;
      const method = request.method.toUpperCase();

      if (method === "GET" && path === "/health") {
        return json({ ok: true, service: "motionscore-license-api" }, 200, env);
      }

      if (method === "POST" && path === "/signup") {
        const body = await requireJson(request);
        const name = asNonEmpty(body.name);
        const institution = asNonEmpty(body.institution);
        const email = normalizeEmail(asNonEmpty(body.email));

        const nowIso = new Date().toISOString();
        const user = await upsertUser(env, {
          name,
          institution,
          email,
          email_verified_at: nowIso, // no-domain MVP: immediate verification
        });
        const license = await getOrCreateActiveLicense(env, user.id);

        await insertEvent(env, {
          license_id: license.id,
          event_type: "signup",
          payload_json: {
            mode: "no-domain-mvp",
            user_id: user.id,
            email,
          },
        });

        return json(
          {
            ok: true,
            mode: "no-domain-mvp",
            user: {
              id: user.id,
              name: user.name,
              institution: user.institution,
              email: user.email,
            },
            license: {
              license_key: license.license_key,
              expires_at: license.expires_at,
              max_devices: license.max_devices,
              status: license.status,
            },
          },
          200,
          env,
        );
      }

      if (method === "POST" && path === "/activate") {
        const body = await requireJson(request);
        const email = normalizeEmail(asNonEmpty(body.email));
        const licenseKey = asNonEmpty(body.license_key);
        const deviceHash = asNonEmpty(body.device_hash);

        const user = await findUserByEmail(env, email);
        if (!user) {
          return json({ ok: false, error: "unknown_email" }, 404, env);
        }
        const license = await findActiveLicenseByKey(env, user.id, licenseKey);
        if (!license) {
          return json({ ok: false, error: "invalid_or_inactive_license" }, 403, env);
        }

        const maxDevices = Number.isFinite(Number(license.max_devices))
          ? Number(license.max_devices)
          : parsePositiveInt(env.MAX_DEVICES, DEFAULT_MAX_DEVICES);

        const nowIso = new Date().toISOString();
        const existingDevice = await findDevice(env, license.id, deviceHash);
        if (existingDevice) {
          await updateDeviceLastSeen(env, existingDevice.id, nowIso);
        } else {
          const devices = await listDevicesForLicense(env, license.id);
          if (devices.length >= maxDevices) {
            return json(
              {
                ok: false,
                error: "device_limit_reached",
                max_devices: maxDevices,
              },
              403,
              env,
            );
          }
          await insertDevice(env, {
            license_id: license.id,
            device_hash: deviceHash,
            last_seen_at: nowIso,
          });
        }

        const ttlSeconds = 60 * 60 * 24 * 30; // 30 days
        const modelDecryptKey = asNonEmpty(env.MODEL_MASTER_KEY);
        const token = await createJwt(env.JWT_SECRET, {
          sub: user.id,
          lid: license.id,
          email: user.email,
          iat: Math.floor(Date.now() / 1000),
          exp: Math.floor(Date.now() / 1000) + ttlSeconds,
        });

        await insertEvent(env, {
          license_id: license.id,
          event_type: "activation",
          payload_json: {
            device_hash: deviceHash,
          },
        });

        return json(
          {
            ok: true,
            token,
            token_type: "Bearer",
            expires_in: ttlSeconds,
            model_decrypt_key: modelDecryptKey,
            license: {
              id: license.id,
              expires_at: license.expires_at,
              max_devices: maxDevices,
              status: license.status,
            },
          },
          200,
          env,
        );
      }

      const modelMatch = path.match(/^\/model\/([^/]+)(?:\/(manifest))?$/);
      if (method === "GET" && modelMatch) {
        const version = decodeURIComponent(modelMatch[1]);
        const manifestMode = modelMatch[2] === "manifest";
        const auth = await requireAuth(request, env);
        if (!auth) {
          return json({ ok: false, error: "unauthorized" }, 401, env);
        }
        const prefix = (env.MODEL_OBJECT_PREFIX || "models").replace(/\/+$/, "");
        const key = manifestMode
          ? `${prefix}/${version}.manifest.json`
          : `${prefix}/${version}.enc`;

        const object = await env.MODELS_BUCKET.get(key);
        if (!object) {
          return json({ ok: false, error: "model_not_found", key }, 404, env);
        }

        if (!manifestMode) {
          await insertEvent(env, {
            license_id: auth.licenseId,
            event_type: "download",
            payload_json: {
              version,
              key,
            },
          });
        }

        const headers = new Headers();
        headers.set("content-type", manifestMode ? "application/json; charset=utf-8" : "application/octet-stream");
        headers.set("cache-control", "private, max-age=300");
        if (object.httpEtag) {
          headers.set("etag", object.httpEtag);
        }
        return withCors(new Response(object.body, { status: 200, headers }), env);
      }

      if (method === "POST" && path === "/event") {
        const auth = await requireAuth(request, env);
        if (!auth) {
          return json({ ok: false, error: "unauthorized" }, 401, env);
        }
        const body = await requireJson(request);
        const eventType = asNonEmpty(body.event_type);
        if (!["prediction", "download", "activation", "signup"].includes(eventType)) {
          return json({ ok: false, error: "invalid_event_type" }, 400, env);
        }
        const payload = typeof body.payload === "object" && body.payload !== null ? (body.payload as JsonMap) : {};

        await insertEvent(env, {
          license_id: auth.licenseId,
          event_type: eventType,
          payload_json: payload,
        });
        return json({ ok: true }, 200, env);
      }

      return json({ ok: false, error: "not_found" }, 404, env);
    } catch (error) {
      const message = error instanceof Error ? error.message : "unexpected_error";
      return json({ ok: false, error: message }, 500, env);
    }
  },
};

function parsePositiveInt(raw: string | undefined, fallback: number): number {
  const value = Number(raw);
  if (Number.isFinite(value) && value > 0) {
    return Math.floor(value);
  }
  return fallback;
}

function asNonEmpty(value: unknown): string {
  const out = String(value ?? "").trim();
  if (!out) {
    throw new Error("missing_required_field");
  }
  return out;
}

function normalizeEmail(email: string): string {
  return email.trim().toLowerCase();
}

async function requireJson(request: Request): Promise<JsonMap> {
  let parsed: unknown;
  try {
    parsed = await request.json();
  } catch {
    throw new Error("invalid_json");
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("json_object_required");
  }
  return parsed as JsonMap;
}

function withCors(response: Response, env: Env): Response {
  const headers = new Headers(response.headers);
  headers.set("access-control-allow-origin", env.ALLOWED_ORIGIN || "*");
  headers.set("access-control-allow-methods", "GET,POST,OPTIONS");
  headers.set("access-control-allow-headers", "content-type,authorization");
  headers.set("access-control-max-age", "86400");
  return new Response(response.body, {
    status: response.status,
    headers,
  });
}

function json(payload: unknown, status: number, env: Env): Response {
  return withCors(
    new Response(JSON.stringify(payload), {
      status,
      headers: {
        "content-type": "application/json; charset=utf-8",
      },
    }),
    env,
  );
}

function generateLicenseKey(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(12));
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("").toUpperCase();
  const groups = hex.match(/.{1,4}/g) || [];
  return `MS-${groups.join("-")}`;
}

async function supabaseFetch<T>(env: Env, pathAndQuery: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers || {});
  headers.set("apikey", env.SUPABASE_SERVICE_ROLE_KEY);
  headers.set("authorization", `Bearer ${env.SUPABASE_SERVICE_ROLE_KEY}`);
  if (!headers.has("content-type") && init?.body) {
    headers.set("content-type", "application/json");
  }

  const response = await fetch(`${env.SUPABASE_URL}/rest/v1${pathAndQuery}`, {
    ...init,
    headers,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`supabase_error:${response.status}:${detail}`);
  }
  if (response.status === 204) {
    return [] as unknown as T;
  }

  // Some Supabase writes return 201 with empty body (or no JSON content-type).
  const raw = await response.text();
  if (!raw || !raw.trim()) {
    return [] as unknown as T;
  }
  try {
    return JSON.parse(raw) as T;
  } catch {
    return [] as unknown as T;
  }
}

async function upsertUser(
  env: Env,
  payload: { name: string; institution: string; email: string; email_verified_at: string },
): Promise<UserRow> {
  const rows = await supabaseFetch<UserRow[]>(
    env,
    `/users?on_conflict=email`,
    {
      method: "POST",
      headers: {
        Prefer: "resolution=merge-duplicates,return=representation",
      },
      body: JSON.stringify(payload),
    },
  );
  if (!rows.length) {
    throw new Error("failed_user_upsert");
  }
  return rows[0];
}

async function findUserByEmail(env: Env, email: string): Promise<UserRow | null> {
  const rows = await supabaseFetch<UserRow[]>(
    env,
    `/users?email=eq.${encodeURIComponent(email)}&select=id,name,institution,email,email_verified_at,created_at&limit=1`,
  );
  return rows[0] || null;
}

async function getOrCreateActiveLicense(env: Env, userId: string): Promise<LicenseRow> {
  const nowIso = new Date().toISOString();
  const existing = await supabaseFetch<LicenseRow[]>(
    env,
    `/licenses?user_id=eq.${encodeURIComponent(userId)}&status=eq.active&expires_at=gt.${encodeURIComponent(nowIso)}&select=id,user_id,license_key,status,max_devices,expires_at,created_at&order=created_at.desc&limit=1`,
  );
  if (existing.length > 0) {
    return existing[0];
  }

  const maxDevices = parsePositiveInt(env.MAX_DEVICES, DEFAULT_MAX_DEVICES);
  const durationDays = parsePositiveInt(env.LICENSE_DURATION_DAYS, DEFAULT_LICENSE_DURATION_DAYS);
  const expiresAt = new Date(Date.now() + durationDays * 24 * 60 * 60 * 1000).toISOString();
  const insert = await supabaseFetch<LicenseRow[]>(
    env,
    `/licenses`,
    {
      method: "POST",
      headers: {
        Prefer: "return=representation",
      },
      body: JSON.stringify({
        user_id: userId,
        license_key: generateLicenseKey(),
        status: "active",
        max_devices: maxDevices,
        expires_at: expiresAt,
      }),
    },
  );
  if (!insert.length) {
    throw new Error("failed_license_insert");
  }
  return insert[0];
}

async function findActiveLicenseByKey(env: Env, userId: string, licenseKey: string): Promise<LicenseRow | null> {
  const nowIso = new Date().toISOString();
  const rows = await supabaseFetch<LicenseRow[]>(
    env,
    `/licenses?user_id=eq.${encodeURIComponent(userId)}&license_key=eq.${encodeURIComponent(licenseKey)}&status=eq.active&expires_at=gt.${encodeURIComponent(nowIso)}&select=id,user_id,license_key,status,max_devices,expires_at,created_at&limit=1`,
  );
  return rows[0] || null;
}

type DeviceRow = {
  id: string;
  license_id: string;
  device_hash: string;
  last_seen_at: string;
  created_at: string;
};

async function findDevice(env: Env, licenseId: string, deviceHash: string): Promise<DeviceRow | null> {
  const rows = await supabaseFetch<DeviceRow[]>(
    env,
    `/devices?license_id=eq.${encodeURIComponent(licenseId)}&device_hash=eq.${encodeURIComponent(deviceHash)}&select=id,license_id,device_hash,last_seen_at,created_at&limit=1`,
  );
  return rows[0] || null;
}

async function listDevicesForLicense(env: Env, licenseId: string): Promise<DeviceRow[]> {
  return supabaseFetch<DeviceRow[]>(
    env,
    `/devices?license_id=eq.${encodeURIComponent(licenseId)}&select=id,license_id,device_hash,last_seen_at,created_at`,
  );
}

async function updateDeviceLastSeen(env: Env, deviceId: string, nowIso: string): Promise<void> {
  await supabaseFetch<unknown>(
    env,
    `/devices?id=eq.${encodeURIComponent(deviceId)}`,
    {
      method: "PATCH",
      body: JSON.stringify({
        last_seen_at: nowIso,
      }),
    },
  );
}

async function insertDevice(
  env: Env,
  payload: { license_id: string; device_hash: string; last_seen_at: string },
): Promise<void> {
  await supabaseFetch<unknown>(
    env,
    `/devices`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  );
}

async function insertEvent(
  env: Env,
  payload: { license_id: string | null; event_type: string; payload_json: JsonMap },
): Promise<void> {
  await supabaseFetch<unknown>(
    env,
    `/events`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  );
}

type AuthPayload = {
  userId: string;
  licenseId: string;
  email: string;
};

async function requireAuth(request: Request, env: Env): Promise<AuthPayload | null> {
  const authHeader = request.headers.get("authorization") || "";
  const token = authHeader.startsWith("Bearer ") ? authHeader.slice("Bearer ".length).trim() : "";
  if (!token) {
    return null;
  }
  const payload = await verifyJwt(env.JWT_SECRET, token);
  if (!payload) {
    return null;
  }
  const userId = String(payload.sub || "").trim();
  const licenseId = String(payload.lid || "").trim();
  const email = String(payload.email || "").trim();
  if (!userId || !licenseId || !email) {
    return null;
  }
  return { userId, licenseId, email };
}

async function createJwt(secret: string, payload: JsonMap): Promise<string> {
  const header = {
    alg: "HS256",
    typ: "JWT",
  };
  const headerPart = base64UrlEncodeUtf8(JSON.stringify(header));
  const payloadPart = base64UrlEncodeUtf8(JSON.stringify(payload));
  const signingInput = `${headerPart}.${payloadPart}`;
  const signature = await hmacSha256Utf8(secret, signingInput);
  const signaturePart = base64UrlEncodeBytes(signature);
  return `${signingInput}.${signaturePart}`;
}

async function verifyJwt(secret: string, token: string): Promise<JsonMap | null> {
  const parts = token.split(".");
  if (parts.length !== 3) {
    return null;
  }
  const [headerPart, payloadPart, signaturePart] = parts;
  const signingInput = `${headerPart}.${payloadPart}`;
  const expectedSig = base64UrlEncodeBytes(await hmacSha256Utf8(secret, signingInput));
  if (expectedSig !== signaturePart) {
    return null;
  }
  let payload: JsonMap;
  try {
    payload = JSON.parse(base64UrlDecodeUtf8(payloadPart)) as JsonMap;
  } catch {
    return null;
  }
  const exp = Number(payload.exp);
  if (!Number.isFinite(exp)) {
    return null;
  }
  const nowSec = Math.floor(Date.now() / 1000);
  if (nowSec >= exp) {
    return null;
  }
  return payload;
}

async function hmacSha256Utf8(secret: string, data: string): Promise<Uint8Array> {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    {
      name: "HMAC",
      hash: "SHA-256",
    },
    false,
    ["sign"],
  );
  const signature = await crypto.subtle.sign("HMAC", key, enc.encode(data));
  return new Uint8Array(signature);
}

function base64UrlEncodeUtf8(input: string): string {
  const bytes = new TextEncoder().encode(input);
  return base64UrlEncodeBytes(bytes);
}

function base64UrlDecodeUtf8(input: string): string {
  const base64 = input.replace(/-/g, "+").replace(/_/g, "/").padEnd(Math.ceil(input.length / 4) * 4, "=");
  const bin = atob(base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    bytes[i] = bin.charCodeAt(i);
  }
  return new TextDecoder().decode(bytes);
}

function base64UrlEncodeBytes(bytes: Uint8Array): string {
  let bin = "";
  for (let i = 0; i < bytes.length; i++) {
    bin += String.fromCharCode(bytes[i]);
  }
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}
