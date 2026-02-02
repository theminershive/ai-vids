# Rollout Checklist

1) Environment
- Verify `.env` has monitoring and timeout settings.
- Ensure `MONITOR_STATUS_DIR` is shared across repos.

2) Monitoring stack
```
cd /home/trilobyte/ai/monitor
docker compose up -d --build
```
Grafana: http://localhost:3000 (admin/admin)

3) Health server
```
python3 monitor_server.py
```
Or systemd:
```
sudo cp systemd/monitor.service /etc/systemd/system/dailybible-monitor.service
sudo systemctl daemon-reload
sudo systemctl enable --now dailybible-monitor.service
```

4) Pipeline scheduler
```
sudo cp systemd/pipeline.service /etc/systemd/system/dailybible.service
sudo cp systemd/pipeline.timer /etc/systemd/system/dailybible.timer
sudo systemctl daemon-reload
sudo systemctl enable --now dailybible.timer
```

5) Verify
- `/health` returns status
- Grafana dashboard shows repo counters
- Run a manual pipeline once

6) Comment bot service (optional)
```
sudo cp systemd/comment.service /etc/systemd/system/dailybible-comment.service
sudo systemctl daemon-reload
sudo systemctl enable --now dailybible-comment.service
```

7) Scheduler service (if repo includes scheduler.py)
```
sudo cp systemd/scheduler.service /etc/systemd/system/dailybible-scheduler.service
sudo systemctl daemon-reload
sudo systemctl enable --now dailybible-scheduler.service
```
