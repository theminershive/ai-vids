#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

from prometheus_client import Gauge, start_http_server
from datetime import datetime

STATUS_DIR = Path(os.getenv("MONITOR_STATUS_DIR", "/home/trilobyte/ai/monitor/status"))
PORT = int(os.getenv("MONITOR_EXPORTER_PORT", "9108"))
POLL_INTERVAL = float(os.getenv("MONITOR_EXPORTER_POLL", "5"))

runs_total = Gauge("pipeline_runs_total", "Total runs", ["repo"])
runs_success = Gauge("pipeline_runs_success", "Successful runs", ["repo"])
runs_error = Gauge("pipeline_runs_error", "Errored runs", ["repo"])
run_status = Gauge("pipeline_run_status", "Current run status", ["repo", "status"])
last_update = Gauge("pipeline_last_update_timestamp_seconds", "Last update timestamp", ["repo"])
last_topic = Gauge("pipeline_last_topic_info", "Last topic info", ["repo", "topic", "run_id", "script"])
last_reference = Gauge("pipeline_last_reference_info", "Last reference info", ["repo", "reference", "run_id"])
section_count = Gauge("pipeline_last_section_count", "Last section count", ["repo"])
last_final = Gauge("pipeline_last_final_info", "Last final video", ["repo", "final_video", "run_id"])
last_run_number = Gauge("pipeline_last_run_number", "Last run number", ["repo"])
scheduler_last_run_ts = Gauge("pipeline_scheduler_last_run_timestamp_seconds", "Scheduler last run", ["repo"])
scheduler_next_run_ts = Gauge("pipeline_scheduler_next_run_timestamp_seconds", "Scheduler next run", ["repo"])
scheduler_last_status = Gauge("pipeline_scheduler_last_status", "Scheduler last status", ["repo", "status"])

stage_elapsed = Gauge("pipeline_stage_last_elapsed_seconds", "Last stage elapsed", ["repo", "stage"])
stage_end_ts = Gauge("pipeline_stage_last_end_timestamp_seconds", "Last stage end timestamp", ["repo", "stage"])

_last_topic_labels: Dict[str, Tuple[str, str, str]] = {}
_last_reference_labels: Dict[str, Tuple[str, str]] = {}
_last_final_labels: Dict[str, Tuple[str, str]] = {}


def _safe_label(value: str) -> str:
    if value is None:
        return ""
    text = str(value)
    return text[:120]


def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_iso_ts(value: str) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return None


def _set_last_topic(repo: str, topic: str, run_id: str, script: str) -> None:
    key = repo
    prev = _last_topic_labels.get(key)
    label = (_safe_label(topic), _safe_label(run_id), _safe_label(script))
    if prev and prev != label:
        last_topic.remove(repo, prev[0], prev[1], prev[2])
    last_topic.labels(repo, label[0], label[1], label[2]).set(1)
    _last_topic_labels[key] = label


def _set_last_reference(repo: str, reference: str, run_id: str) -> None:
    key = repo
    prev = _last_reference_labels.get(key)
    label = (_safe_label(reference), _safe_label(run_id))
    if prev and prev != label:
        last_reference.remove(repo, prev[0], prev[1])
    last_reference.labels(repo, label[0], label[1]).set(1)
    _last_reference_labels[key] = label


def _set_last_final(repo: str, final_video: str, run_id: str) -> None:
    key = repo
    prev = _last_final_labels.get(key)
    label = (_safe_label(final_video), _safe_label(run_id))
    if prev and prev != label:
        last_final.remove(repo, prev[0], prev[1])
    last_final.labels(repo, label[0], label[1]).set(1)
    _last_final_labels[key] = label


def update_metrics() -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    for path in STATUS_DIR.glob("*.json"):
        state = _load_state(path)
        repo = state.get("repo") or path.stem

        counters = state.get("counters", {})
        runs_total.labels(repo).set(counters.get("runs_total", 0))
        runs_success.labels(repo).set(counters.get("runs_success", 0))
        runs_error.labels(repo).set(counters.get("runs_error", 0))

        status = (state.get("run", {}) or {}).get("status")
        for st in ("running", "success", "error"):
            run_status.labels(repo, st).set(1 if status == st else 0)

        last_update_epoch = state.get("last_update_epoch") or 0
        last_update.labels(repo).set(last_update_epoch)

        last = state.get("last", {}) or {}
        topic = last.get("topic", "")
        run_id = (state.get("run", {}) or {}).get("id") or last.get("run_id") or ""
        script_path = last.get("script_path", "")
        if topic or run_id or script_path:
            _set_last_topic(repo, topic, run_id, script_path)

        reference = last.get("reference", "")
        if reference or run_id:
            _set_last_reference(repo, reference, run_id)

        final_video = last.get("final_video", "")
        if final_video or run_id:
            _set_last_final(repo, final_video, run_id)

        run_number = last.get("run_number")
        if run_number is not None:
            try:
                last_run_number.labels(repo).set(float(run_number))
            except Exception:
                last_run_number.labels(repo).set(0)

        sched_last = last.get("scheduler_last_run")
        sched_next = last.get("scheduler_next_run")
        sched_status = last.get("scheduler_last_status")
        ts_last = _parse_iso_ts(sched_last) if isinstance(sched_last, str) else None
        ts_next = _parse_iso_ts(sched_next) if isinstance(sched_next, str) else None
        if ts_last is not None:
            scheduler_last_run_ts.labels(repo).set(ts_last)
        if ts_next is not None:
            scheduler_next_run_ts.labels(repo).set(ts_next)
        if sched_status:
            for st in ("Success", "Failure", "Never Run", "Timeout", "Error"):
                scheduler_last_status.labels(repo, st).set(1 if sched_status == st else 0)

        if "section_count" in last:
            try:
                section_count.labels(repo).set(float(last.get("section_count") or 0))
            except Exception:
                section_count.labels(repo).set(0)

        for stage, entry in (state.get("stages", {}) or {}).items():
            elapsed = entry.get("last_elapsed_sec")
            if elapsed is not None:
                stage_elapsed.labels(repo, stage).set(float(elapsed))
            end_epoch = entry.get("last_end_epoch")
            if end_epoch is not None:
                stage_end_ts.labels(repo, stage).set(float(end_epoch))


if __name__ == "__main__":
    start_http_server(PORT)
    print(f"[monitor] exporter listening on :{PORT}, status_dir={STATUS_DIR}")
    while True:
        update_metrics()
        time.sleep(POLL_INTERVAL)
