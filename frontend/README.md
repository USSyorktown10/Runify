# Runify Frontend

Vite + React + TypeScript SPA for the Runify API.

## Quick start

```bash
# From repo root — start API + DB
docker compose up postgres web -d

# Frontend (local)
cd frontend
cp .env.example .env   # uses /api proxy to localhost:8000
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

## Docker (API + frontend)

```bash
docker compose up --build
```

Frontend: `http://localhost:5173` · API: `http://localhost:8000`

## Design docs

- [docs/frontend/VIEWS.md](../docs/frontend/VIEWS.md) — view catalog
- [docs/frontend/UI_LANGUAGE.md](../docs/frontend/UI_LANGUAGE.md) — design system

## Seeded demo account

After running `scripts/seed_bulk.py`, log in with **demo** / **demo1234**.

## API types

Types are hand-maintained in `src/types/api.ts` aligned with `RUNIFY_R1.md`. When the backend OpenAPI schema is stable, regenerate with:

```bash
npx openapi-typescript http://localhost:8000/openapi.json -o src/types/openapi.d.ts
```

## Stack

- React 19, React Router 7, TanStack Query
- Tailwind CSS 4 (Astro Cactus–inspired tokens)
- Leaflet + Recharts for maps and performance charts
