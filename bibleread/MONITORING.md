# Monitoring

This repo writes status updates to the shared monitor folder:
- Status JSON: `/home/trilobyte/ai/monitor/status/bibleread.json`

## Health endpoint
Start the lightweight health server:
```
python3 monitor_server.py
```
Defaults to `MONITOR_HEALTH_PORT` in `.env`.

Example:
```
curl http://localhost:$MONITOR_HEALTH_PORT/health
curl http://localhost:$MONITOR_HEALTH_PORT/status
```

## Grafana dashboard
The shared stack lives in `/home/trilobyte/ai/monitor`:
```
cd /home/trilobyte/ai/monitor
docker compose up -d --build
```
Grafana: http://localhost:3000 (admin/admin)

## Metrics emitted
- runs_total / runs_success / runs_error
- current run status
- last topic / reference / final video / run number
- per-stage timing

## Health systemd service
```
sudo cp systemd/monitor.service /etc/systemd/system/bibleread-monitor.service
sudo systemctl daemon-reload
sudo systemctl enable --now bibleread-monitor.service
```

Scheduler status (if `scheduler.py` exists) is included in `/health` and `/status`.
