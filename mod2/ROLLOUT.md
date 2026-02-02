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
sudo cp systemd/monitor.service /etc/systemd/system/mod2-monitor.service
sudo systemctl daemon-reload
sudo systemctl enable --now mod2-monitor.service
```

4) Pipeline scheduler
```
sudo cp systemd/pipeline.service /etc/systemd/system/mod2.service
sudo cp systemd/pipeline.timer /etc/systemd/system/mod2.timer
sudo systemctl daemon-reload
sudo systemctl enable --now mod2.timer
```

5) Verify
- `/health` returns status
- Grafana dashboard shows repo counters
- Run a manual pipeline once

6) Comment bot service (optional)
```
sudo cp systemd/comment.service /etc/systemd/system/mod2-comment.service
sudo systemctl daemon-reload
sudo systemctl enable --now mod2-comment.service
```

7) Scheduler service
- Not applicable (this repo does not include `scheduler.py`).

8) Comment bot
```
python3 comment.py
```

9) Uploaders
```
python3 ytuploader.py <path_to_json>
python3 fbupload.py <path_to_json>
python3 igupload.py <path_to_json>
```
