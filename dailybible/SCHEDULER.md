# Scheduler

This repo includes a systemd service + timer for running the pipeline on a schedule.

Files:
- `systemd/pipeline.service`
- `systemd/pipeline.timer`

## Install
```
sudo cp systemd/pipeline.service /etc/systemd/system/dailybible.service
sudo cp systemd/pipeline.timer /etc/systemd/system/dailybible.timer
sudo systemctl daemon-reload
sudo systemctl enable --now dailybible.timer
```

## Change the schedule
Edit `/etc/systemd/system/dailybible.timer` and adjust the `[Timer]` section:

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
sudo systemctl restart dailybible.timer
```

## Run once manually
```
python3 app.py
```
(For `mod`, run `python3 pipeline.py`.)

## Internal scheduler service (scheduler.py)
```
sudo cp systemd/scheduler.service /etc/systemd/system/dailybible-scheduler.service
sudo systemctl daemon-reload
sudo systemctl enable --now dailybible-scheduler.service
```
