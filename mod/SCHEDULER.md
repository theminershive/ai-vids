# Scheduler

This repo includes a systemd service + timer for running the pipeline on a schedule.

Files:
- `systemd/pipeline.service`
- `systemd/pipeline.timer`

## Install
```
sudo cp systemd/pipeline.service /etc/systemd/system/mod.service
sudo cp systemd/pipeline.timer /etc/systemd/system/mod.timer
sudo systemctl daemon-reload
sudo systemctl enable --now mod.timer
```

## Change the schedule
Edit `/etc/systemd/system/mod.timer` and adjust the `[Timer]` section:

- Every 15 minutes:
  `OnUnitActiveSec=15min`
- Every hour:
  `OnUnitActiveSec=1h`
- Twice per day:
  `OnCalendar=*-*-* 09:00:00`
  `OnCalendar=*-*-* 21:00:00`

After changes:
```
sudo systemctl daemon-reload
sudo systemctl restart mod.timer
```

## Run once manually
```
python3 pipeline.py
```
