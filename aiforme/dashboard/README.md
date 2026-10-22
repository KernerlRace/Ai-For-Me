# Dashboard

The assistant's local web UI, started early in Phase 2 for the calendar. FastAPI serves a JSON API and the built React
app on one port, http://127.0.0.1:8765 — local only, no accounts.

| Page | What it shows |
|---|---|
| Calendar | Day, week and month views in Russian; drag over empty time to create, drag or stretch to move, click to edit; events coloured by kind — meeting, shift, other |
| Header | The mode the [autopilot](../mode_autopilot) is in, and why on hover |

Planned as their phases land: the log of what the assistant did and why, corrections for the active-learning loop,
finance and learning widgets, security alerts.

```bash
cd aiforme/dashboard/web
npm install
npm run build            # dist/, served by the API
cd ../../..
uv run python -m aiforme dashboard
```

`npm run dev` serves the frontend with hot reload on :5173 and proxies `/api` to the backend. The API is documented at
`/api/docs`; every calendar change is also published on the bus as `calendar.changed`.
