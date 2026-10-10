# CivicLens

A civic complaint platform: citizens report local issues (with photos, a map pin, and voice dictation), an AI triage step categorizes and prioritizes each report, and an admin console manages the queue and drafts official response letters.

> **Live:** https://civic-lens-sunaad7.vercel.app · **API:** https://civic-lens-api.vercel.app · [health](https://civic-lens-api.vercel.app/healthz)

The project has three parts:

| Part | Folder | What it does |
|------|--------|--------------|
| Web app | `web/` | Citizen app and admin console (React + Vite) |
| API | `api/` | Auth, complaints, admin workflow, letters (Express + PostGIS) |
| ML service | `civiclens-ml/` | AI triage: detection, severity, priority, repair-time, clustering, hotspots (FastAPI) |

## Screenshots

### Landing page

| Hero | Features | How it works |
|---|---|---|
| ![Landing hero](docs/screenshots/landing.png) | ![Landing features](docs/screenshots/landing2.png) | ![Landing how it works](docs/screenshots/landing3.png) |

### App

| Complaint with AI triage (ML service) | User dashboard |
|---|---|
| ![AI triage](docs/screenshots/triage.png) | ![Dashboard](docs/screenshots/dashboard.png) |

## Architecture

```
┌──────────────────────────┐
│  React + Vite (Vercel)   │
│  Citizen app │ Admin UI  │
│  Leaflet · Web Speech    │
└────────────┬─────────────┘
             │ REST (JSON, JWT)
┌────────────▼─────────────┐
│  Node/Express (Vercel)   │
│  ├─ auth                 │
│  ├─ complaints           │
│  ├─ admin                │
│  └─ ai service ──────────┼──► civiclens-ml (FastAPI)
│                          │      image → detection, severity,
│                          │      priority, ETA, materials
└───────┬──────────┬───────┘
        │          │
┌───────▼───────┐ ┌▼──────────────────┐
│ Postgres +    │ │ Object storage    │
│ PostGIS       │ │ (Supabase Storage)│
│ (Supabase)    │ │ complaint images  │
└───────────────┘ └───────────────────┘
```

## Stack

| Layer     | Tech |
|-----------|------|
| Frontend  | React 19, Vite 8, TypeScript, react-router 8, TanStack Query, Leaflet / react-leaflet, Web Speech API |
| Backend   | Node.js, Express 5, TypeScript, Zod, Kysely, JWT (access + rotating refresh cookies) |
| Database  | PostgreSQL + PostGIS (Supabase in production, Docker locally) |
| Storage   | Supabase Storage (private bucket, signed URLs) — falls back to local disk when unconfigured |
| AI (API)  | Provider interface with a built-in **stub** (no API key needed) |
| ML service | Python, FastAPI, YOLOv8 (Ultralytics), scikit-learn, OpenCV, DBSCAN |
| Tests     | Vitest + Supertest (API), Vitest + Testing Library (web), pytest (ML) |

## Repo layout

```
├── web/          # Citizen + admin UI (Vercel, root directory: web)
├── api/          # Express API (Vercel, root directory: api — also runs as a long-lived server)
├── civiclens-ml/ # FastAPI ML triage service (detection, severity, ETA, clustering)
├── migrations/   # (in api/migrations) plain SQL migrations
├── docker-compose.yml
└── PLAN.md       # the implementation plan this was built from
```

## ML service (`civiclens-ml/`)

A standalone FastAPI microservice that the Node/Express backend calls over HTTP to triage each reported issue (potholes, garbage, damaged streetlights, waterlogging).

**What it does**

- **Photo validation (EXIF):** flags fake, stale or wrong-location photos, plus perceptual-hash re-use checks
- **Detection:** YOLOv8 object detector with a mock fallback for demos
- **Severity 1-5:** explainable, rule-based (area + class + count), returned with a list of reasons
- **Priority score:** uses nearby hospitals, schools and roads
- **Resolution plan:** repair-time prediction (RandomForest) plus materials, tools and crew size
- **Duplicate merging:** DBSCAN clustering of nearby reports into master incidents
- **Hotspots:** heatmap grid with a 7-day trend forecast
- **Complaint classifier:** English / Tamil / Hindi text to category and urgency
- **Trust score:** dynamic reporter trust and tier
- **After-photo check:** `/verify-repair` confirms a fix before the ticket is closed
- **Privacy:** face and number-plate blurring
- **Human in the loop:** low-confidence results return `NEEDS_REVIEW` instead of auto-dispatching

**Run it**

```bash
cd civiclens-ml
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- API docs: http://localhost:8000/docs
- Demo UI: http://localhost:8000/demo
- Tests: `pip install -r requirements-dev.txt && pytest -q`

**Connect it to the API:** deploy or run the service, set `ML_SERVICE_URL` and `ML_API_KEY` for the backend, and call it with `civiclens-ml/integration/node_client.js`. The `integration/ai_triage.sql` file holds the DB table for storing triage results.

**Example endpoints**

| Method | Path | Purpose |
|---|---|---|
| POST | `/analyze` | Main triage: image + GPS/time/context → status, issue, severity, priority, ETA, materials, explanation |
| POST | `/verify-repair` | After-photo check |
| POST | `/cluster` | Duplicate merging |
| POST | `/hotspots` | Hotspot grid + forecast |
| POST | `/complaint/classify` | Multilingual complaint classification |
| POST | `/trust-score` | Reporter trust score |
| GET | `/health`, `/model/info` | Ops and model metrics |

### Honest status and limitations

- **Mock mode:** without trained weights (`civiclens-ml/models/best.pt`) the service runs in **mock mode**. The whole pipeline works end to end, but detection is a deterministic placeholder. Every response carries `"model_mode": "mock"` or `"yolo"`, and the UI shows "Demo detection (no trained model loaded)" in mock mode. Mock results are not real detection.
- **Hosting:** the live Vercel deployment runs the API with the built-in stub AI provider. The ML service is a separate Python service (run locally or with Docker; it is not part of the Vercel projects).
- The ETA model is trained on **simulated** data, since no public repair-time dataset exists.
- Severity is rule-based, not a trained regressor.
- Complaint NLP is a keyword baseline, not a fine-tuned language model.
- Face and plate blur use Haar cascades, which will miss some faces and plates.
- Re-use detection keeps hashes in memory (resets on restart).

See [`civiclens-ml/README.md`](civiclens-ml/README.md) and [`civiclens-ml/MODEL_CARD.md`](civiclens-ml/MODEL_CARD.md) for full details, and `civiclens-ml/training/colab_train.md` for training the YOLO model.

## Local development

Prerequisites: Node.js 22.19+ (24 LTS recommended), npm 11+, Docker (or any Postgres 15+ with PostGIS). The ML service additionally needs Python 3.10+.

```bash
# 1. Start Postgres + PostGIS
docker compose up -d

# 2. Configure both apps
cp api/.env.example api/.env          # edit if your DB differs
cp web/.env.example web/.env          # optional, defaults are fine

# 3. Install dependencies
npm install
npm install --prefix api
npm install --prefix web

# 4. Create schema + seeded admin user
npm run migrate --prefix api
npm run seed   --prefix api

# 5. Run both apps (API :3001, web :5173 with proxy)
npm run dev
```

Sign in with the seeded admin from `api/.env` (defaults: `admin@civiclens.local` / `admin1234`).

> No Docker? Point `api/.env` `DATABASE_URL` at any Postgres that has the `postgis` and `citext` extensions available, then run the migrate step.

To also run the ML service, start it in a second terminal as shown in the [ML service](#ml-service-civiclens-ml) section.

### Scripts

| Where | Command | Does |
|-------|---------|------|
| root  | `npm run dev` | API + web concurrently |
| root  | `npm run build` / `test` / `typecheck` | run across both apps |
| api   | `npm run dev` | Express with hot reload (tsx watch) |
| api   | `npm run migrate` / `seed` / `db:types` | database tasks (`db:types` regenerates Kysely types) |
| api   | `npm test` | Vitest suite (spins migrations on `civiclens_test`) |
| web   | `npm run dev` / `build` / `test` / `lint` | Vite tasks |
| civiclens-ml | `pytest -q` | ML service tests |

## Environment variables

### `api/.env`

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | ✅ | Postgres connection string (PostGIS required) |
| `JWT_ACCESS_SECRET` | ✅ | ≥16 chars locally, **≥32 in production** |
| `JWT_REFRESH_SECRET` | ✅ | ≥16 chars locally, **≥32 in production**, must differ from the access secret |
| `PORT` | — | default `3001` |
| `CORS_ORIGINS` | — | comma-separated allowlist, default `http://localhost:5173` |
| `TRUST_PROXY` | — | reverse-proxy hops to trust for client IP, default `1` in production / `0` otherwise |
| `COOKIE_SAMESITE` | — | refresh-cookie `SameSite`, default `lax` in dev and `none` in production (needed when the web app is on a different site than the API) |
| `AI_PROVIDER` | — | `stub` (default). Add new providers in `src/modules/ai/` |
| `ML_SERVICE_URL` / `ML_API_KEY` | — | base URL and key of the `civiclens-ml` service, used when connecting the ML service |
| `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` | — | if both set → Supabase Storage, else local disk (`UPLOAD_DIR`) |
| `SUPABASE_BUCKET` | — | default `complaint-images` (create it as a **private** bucket) |
| `API_PUBLIC_URL` | prod ✅ | public base URL of this API, used for local-disk image URLs |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | seed | admin account created by `npm run seed` |

### `web/.env`

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_BASE_URL` | `/api` | change only if API is on another origin |
| `VITE_MAP_CENTER` | `12.9716,77.5946` | initial map center as `lat,lng` |

### `civiclens-ml/.env`

See `civiclens-ml/.env.example`. Main settings: `ML_API_KEY`, `CORS_ORIGINS`, `ALLOWED_IMAGE_HOSTS`.

## API overview

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/auth/register` | — | create account → access token + refresh cookie |
| POST | `/api/auth/login` | — | sign in |
| POST | `/api/auth/refresh` | cookie | rotate refresh token, new access token |
| POST | `/api/auth/logout` | cookie | revoke refresh token |
| GET  | `/api/auth/me` | bearer | current user |
| PATCH | `/api/auth/me` | bearer | update name / email / password (email + password changes need `current_password`) |
| POST | `/api/complaints` | bearer | create complaint (multipart: fields + up to 5 `images`) |
| GET  | `/api/complaints` | bearer | my complaints — or `?bbox=w,s,e,n[&status=]` for the map |
| GET  | `/api/complaints/:id` | bearer | detail (owner or admin): images, AI triage, history |
| GET  | `/api/admin/complaints` | admin | queue, filters `?status=&q=` (`q` matches title, description or complaint ID) |
| PATCH | `/api/admin/complaints/:id/status` | admin | transition status (validated) + note |
| POST | `/api/admin/complaints/:id/assign` | admin | assign to an admin user |
| GET  | `/api/admin/complaints/:id/letter` | admin | draft/cache official letter |
| GET  | `/api/admin/users` | admin | admin users for assignment |
| GET  | `/healthz` | — | liveness probe (process + config) |
| GET  | `/readyz` | — | readiness probe (pings the database) |

Complaint lifecycle: `submitted → triaged → assigned → in_progress → resolved` (with `rejected` exits; the server enforces valid transitions).

## AI triage & letters

The AI layer is a small provider interface (`api/src/modules/ai/provider.ts`):

```ts
interface AiProvider {
  triage(input): Promise<TriageResult>       // category, severity 1-5, department, confidence, summary
  draftLetter(input): Promise<string>
}
```

The default `stub` provider is deterministic and offline. To integrate a real model or the ML service:

1. Add your SDK or HTTP client + a provider file (e.g. `civiclens-ml.ts`, using `civiclens-ml/integration/node_client.js`) implementing `AiProvider`.
2. Extend the `AI_PROVIDER` enum in `src/config.ts` and the switch in `src/modules/ai/service.ts`.
3. Set `AI_PROVIDER` + the service URL and API key env vars.

Triage runs inline when a complaint is created; the letter is generated on first request and cached on the complaint row.

## Deployment

The web app and API deploy to Vercel as **two projects** from this one repo. The ML service is a separate Python service (see [ML service](#ml-service-civiclens-ml)); a Dockerfile is included for hosts such as Hugging Face Spaces, Render or Railway.

| App | Live URL | Vercel project | Root directory |
|-----|----------|----------------|----------------|
| Web SPA | https://civic-lens-sunaad7.vercel.app | `civic-lens` | `web` |
| API | https://civic-lens-api.vercel.app | `civic-lens-api` | `api` |

The API runs as a serverless function — Vercel's Node/Express runtime uses `api/src/app.ts` as the function entry, which default-exports the Express app; `api/src/index.ts` imports that same instance and adds the long-lived listener + graceful shutdown when it isn't running on Vercel (local, Docker, or any Node host).

**Web → Vercel:** new project with **root directory `web`** (build `npm run build`, output `dist`). `web/vercel.json` provides SPA rewrites plus security and asset-cache headers. Set `VITE_API_BASE_URL` to the deployed API URL including the `/api` suffix (e.g. `https://civic-lens-api.vercel.app/api`); the default `/api` requires a same-origin rewrite/proxy.

**API → Vercel:** new project with **root directory `api`** (no build command needed — Vercel builds the Express app automatically). Set the environment variables below, including `CORS_ORIGINS` (exact web origin, e.g. `https://civic-lens-sunaad7.vercel.app`, no trailing slash, comma-separated for multiple) and `API_PUBLIC_URL` (the API's own URL). Run migrations and the seed **locally** against the same `DATABASE_URL` before/after deploy: `npm run migrate --prefix api && npm run seed --prefix api`.

**Serverless notes:** use Supabase's **transaction pooler** connection string (port `6543`) for `DATABASE_URL` so many function instances share a bounded set of connections. Vercel caps request bodies at **4.5 MB**, so the API rejects uploads over 4 MB and the web client compresses photos to JPEG before uploading. Vercel also runs its own TypeScript check and module loader on the API function, so `helmet` and `express-rate-limit` are imported in a resolution-agnostic way in `api/src/app.ts` (named import for the rate limiter; a namespace unwrap for helmet's factory) to satisfy both the build-time type-check and the runtime interop.

**Cross-origin cookies:** when the web app and API live on different sites (web + API on separate Vercel domains), the refresh cookie must be `SameSite=None; Secure`. This is the production default; the web app already sends `credentials: 'include'` (set `COOKIE_SAMESITE=none` explicitly if you override it). Both origins must be HTTPS.

**Storage:** create a **private** bucket named `complaint-images` in Supabase and set `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` on the API project. Without them, images are written to the (ephemeral) local disk — fine for local demos, not for production.

**ML service (Docker):**

```bash
cd civiclens-ml
docker build -t civiclens-ml .
docker run -p 7860:7860 civiclens-ml
```

Set `ML_API_KEY`, `CORS_ORIGINS` and ideally `ALLOWED_IMAGE_HOSTS`. Free hosts sleep when idle, so the first request can take about 30 seconds.

### Production checklist

- [ ] `JWT_ACCESS_SECRET` / `JWT_REFRESH_SECRET` are ≥32 random chars and different (the API refuses to boot otherwise).
- [ ] `ADMIN_PASSWORD` is not the default `admin1234`.
- [ ] `CORS_ORIGINS` lists only your real web origin(s).
- [ ] Supabase Storage configured with a private bucket (images survive restarts).
- [ ] `NODE_ENV=production`, `TRUST_PROXY=1`, `COOKIE_SAMESITE=none` on the API project.
- [ ] `VITE_API_BASE_URL` points at the deployed API.
- [ ] `/healthz` and `/readyz` both return `{ "ok": true }`.
- [ ] ML service: `ML_API_KEY` set and trained `models/best.pt` loaded (`/health` shows `"detector_mode":"yolo"`).

The API also ships with `helmet` security headers, `compression`, and rate limiting (600 req / 15 min per IP overall; 40 / 15 min on `/api/auth`) — all skipped under `NODE_ENV=test`.

## Testing

```bash
npm test --prefix api    # 39 tests: auth (incl. profile), complaints (incl. geo bbox), admin, health
npm test --prefix web    # 13 component tests
cd civiclens-ml && pytest -q   # ML service tests
```

The API suite migrates a separate `civiclens_test` database automatically (derived from `DATABASE_URL`).

## Team

| Area | Contributor |
|------|-------------|
| Frontend, backend API, deployment | [@sunaad7](https://github.com/sunaad7) (Sunaad Joel Rampurapu) |
| ML service (`civiclens-ml/`) | [@chakradar1807](https://github.com/chakradar1807) |
