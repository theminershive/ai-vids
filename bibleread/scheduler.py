# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import time
import subprocess
import json
import os
import threading
import traceback
import logging
from datetime import datetime, timedelta
from testemail import send_email
import random  # For jitter in set times

# === CONFIGURATION ===
SCHEDULE_MODE = "set_time"     # "interval" or "set_time"
RUN_INTERVAL_MINUTES = 5
SET_RUN_TIMES = ["07:49", "11:49", "15:49", "19:49", "23:49", "03:49"]
JITTER_MINUTES = 5  # Allow ±5 minutes jitter for set times

PROGRAM = "app.py"
STATUS_FILE = "scheduler_status.json"
LOG_FILE = "scheduler.log"

EMAIL_ENABLED = True
EMAIL_SUBJECT_PREFIX = "[Scheduler Run Report]"
PROGRAM_TIMEOUT_SECONDS = 3800
FAILURE_ALERT_THRESHOLD = 3
# === END CONFIGURATION ===

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS_PATH = os.path.join(BASE_DIR, STATUS_FILE)
LOG_PATH = os.path.join(BASE_DIR, LOG_FILE)

# --- Logging setup ---
logger = logging.getLogger("scheduler")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)

def load_status():
    now = datetime.now().replace(microsecond=0)
    if os.path.exists(STATUS_PATH):
        try:
            with open(STATUS_PATH, "r", encoding="utf-8") as f:
                status = json.load(f)
            logger.info("Loaded status from disk.")
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.warning(f"Status file corrupted ({e}); resetting.")
            os.remove(STATUS_PATH)
        else:
            try:
                nxt = datetime.fromisoformat(status["next_run"])
                delta = (nxt - now).total_seconds()
            except Exception as e:
                logger.warning(f"Invalid next_run format ({e}); resetting.")
                delta = None

            if delta is None or delta > RUN_INTERVAL_MINUTES * 120 or delta < -60:
                logger.info("Stale or past next_run detected; resetting.")
                status["scheduler_started_time"] = now.isoformat()
                if SCHEDULE_MODE == "interval":
                    status["next_run"] = (now + timedelta(minutes=RUN_INTERVAL_MINUTES)).isoformat()
                else:
                    status["next_run"] = get_next_set_time(now).isoformat()
                status["failure_streak"] = 0
                save_status(status)
            return status

    logger.info("Initializing new status.")
    if SCHEDULE_MODE == "interval":
        next_run = now + timedelta(minutes=RUN_INTERVAL_MINUTES)
    else:
        next_run = get_next_set_time(now)
    return {
        "runs_completed": 0,
        "last_run": None,
        "last_status": "Never Run",
        "last_title": "",
        "next_run": next_run.isoformat(),
        "scheduler_started_time": now.isoformat(),
        "failure_streak": 0
    }

def save_status(status):
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=4)
    logger.info("Saved status to disk.")

def run_program():
    logger.info(f"Launching subprocess: {PROGRAM}")
    try:
        proc = subprocess.Popen(
            ["python3", PROGRAM],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        start = time.time()
        while True:
            line = proc.stdout.readline()
            if line:
                logger.info(f"[{PROGRAM}] {line.rstrip()}")
            if proc.poll() is not None:
                break
            if time.time() - start > PROGRAM_TIMEOUT_SECONDS:
                proc.kill()
                logger.error(f"{PROGRAM} killed after timeout.")
                return False, "Timeout"
        for line in proc.stdout:
            logger.info(f"[{PROGRAM}] {line.rstrip()}")

        ret = proc.wait()
        if ret == 0:
            logger.info(f"{PROGRAM} exited with code 0.")
            return True, "Success"
        else:
            logger.error(f"{PROGRAM} exited with code {ret}.")
            return False, f"Exit {ret}"
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error running {PROGRAM}: {tb}")
        return False, "Unexpected error"

def get_next_set_time(now):
    today = now.date()
    # Determine base run time for today or next day
    base_dt = None
    for t in SET_RUN_TIMES:
        hh, mm = map(int, t.split(":"))
        dt = datetime.combine(today, datetime.min.time()).replace(hour=hh, minute=mm)
        if dt > now:
            base_dt = dt
            break
    if base_dt is None:
        hh, mm = map(int, SET_RUN_TIMES[0].split(":"))
        base_dt = datetime.combine(today + timedelta(days=1), datetime.min.time()).replace(hour=hh, minute=mm)
    # Apply random jitter of ±JITTER_MINUTES
    jitter = random.uniform(-JITTER_MINUTES, JITTER_MINUTES)
    dt_jittered = base_dt + timedelta(minutes=jitter)
    logger.info(f"Next run scheduled at {dt_jittered.strftime('%Y-%m-%d %H:%M:%S')} "
                f"(base {base_dt.strftime('%H:%M')} ± {jitter:.2f}m)")
    return dt_jittered
def send_run_email(subject, body):
    if not EMAIL_ENABLED:
        logger.info("Email disabled; skipping.")
        return
    def _bg():
        try:
            logger.info(f"Email → {subject}")
            send_email(subject, body)
            logger.info("Email sent.")
        except Exception as e:
            logger.error(f"Email error: {e}")
    threading.Thread(target=_bg, daemon=True).start()

def generate_email_body(status):
    return (
        f"Scheduler Run Report:\n\n"
        f"Title:       {status['last_title']}\n"
        f"Result:      {status['last_status']}\n"
        f"Completed:   {status['last_run']}\n"
        f"Next Run:    {status['next_run']}\n\n"
        f"Total Runs:  {status['runs_completed']}\n"
    )

def print_status(status):
    print("\n=== Scheduler Status ===")
    print(f"Runs Completed: {status['runs_completed']}")
    print(f"Last Run:       {status['last_run']}")
    print(f"Last Status:    {status['last_status']}")
    print(f"Next Run:       {status['next_run']}")
    print("=========================\n")

def scheduler_loop():
    status = load_status()
    print_status(status)

    while True:
        now = datetime.now().replace(microsecond=0)
        next_run_time = datetime.fromisoformat(status["next_run"])

        if now >= next_run_time:
            success, msg = run_program()
            now2 = datetime.now().replace(microsecond=0)

            status["runs_completed"] += 1
            status["last_run"] = now2.isoformat()
            status["last_status"] = msg
            status["last_title"] = f"Run #{status['runs_completed']}"

            if SCHEDULE_MODE == "interval":
                nr = now2 + timedelta(minutes=RUN_INTERVAL_MINUTES)
            else:
                nr = get_next_set_time(now2)
            status["next_run"] = nr.isoformat()
            status["scheduler_started_time"] = now2.isoformat()
            save_status(status)
            print_status(status)

            if success:
                status["failure_streak"] = 0
                send_run_email(f"{EMAIL_SUBJECT_PREFIX} {status['last_title']}", generate_email_body(status))
            else:
                status["failure_streak"] += 1
                if status["failure_streak"] >= FAILURE_ALERT_THRESHOLD:
                    send_run_email("[ALERT] Scheduler Failure", f"Failed {status['failure_streak']} times.")

            time.sleep(5)
        else:
            sec = int((next_run_time - now).total_seconds())
            m, s = divmod(sec, 60)
            msg = f"Next run in {m:02d}:{s:02d}"
            print(f"\r{msg}", end="")
            logger.info(msg)
            time.sleep(1)

if __name__ == "__main__":
    scheduler_loop()
