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
sudo cp systemd/monitor.service /etc/systemd/system/bibleread-monitor.service
sudo systemctl daemon-reload
sudo systemctl enable --now bibleread-monitor.service
```

4) Pipeline scheduler
```
sudo cp systemd/pipeline.service /etc/systemd/system/bibleread.service
sudo cp systemd/pipeline.timer /etc/systemd/system/bibleread.timer
sudo systemctl daemon-reload
sudo systemctl enable --now bibleread.timer
```

5) Verify
- `/health` returns status
- Grafana dashboard shows repo counters
- Run a manual pipeline once

6) Comment bot service (optional)
```
sudo cp systemd/comment.service /etc/systemd/system/bibleread-comment.service
sudo systemctl daemon-reload
sudo systemctl enable --now bibleread-comment.service
```

7) Scheduler service (if repo includes scheduler.py)
```
sudo cp systemd/scheduler.service /etc/systemd/system/bibleread-scheduler.service
sudo systemctl daemon-reload
sudo systemctl enable --now bibleread-scheduler.service
```
