# Pipeline Monitoring

This folder hosts a lightweight monitoring stack for the pipeline repos:
- `mod`
- `mod2`
- `bibleread`
- `dailybible`
- `ai-tthub`

Each repo writes status JSON into `./monitor/status/<repo>.json`. A small Prometheus exporter reads those files and exposes metrics. Grafana then renders dashboards.

## Enable monitoring in the repos
Add to each repo `.env` (already applied):
```
MONITOR_ENABLE=1
MONITOR_STATUS_DIR=/home/trilobyte/ai/monitor/status
```

## Run with Docker
From this directory:
```
docker compose up -d --build
```

Services:
- Exporter: http://localhost:9108/metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

Dashboard:
- Pipeline Monitoring (auto-provisioned)

## Run exporter directly (no Docker)
```
python exporter.py
```
Environment options:
- `MONITOR_STATUS_DIR` (default `/home/trilobyte/ai/monitor/status`)
- `MONITOR_EXPORTER_PORT` (default `9108`)
- `MONITOR_EXPORTER_POLL` (default `5` seconds)

## What gets tracked
- Run totals / success / errors
- Current run status per repo
- Last topic/reference/final video
- Stage timings (visuals, tts, assemble, captions, overlay)
- Scheduler status (last/next run) when `scheduler.py` is present

## Health endpoints
Each repo includes a small HTTP server:
```
python3 monitor_server.py
```
Defaults to the repo’s `MONITOR_HEALTH_PORT`.

Health payloads include comment bot state (if `fb_user_thread_state.json` exists).

## Status file format
Each repo writes a JSON file containing:
- `counters`: total/success/error
- `run`: current run state
- `stages`: last timing per stage
- `last`: last topic/reference/final video/section_count

The exporter reads these fields and exposes Prometheus metrics.
