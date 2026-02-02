import json
import os
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional


def _now() -> tuple[float, str]:
    epoch = time.time()
    iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))
    return epoch, iso


def _default_state(repo: str) -> Dict[str, Any]:
    return {
        "repo": repo,
        "last_update": None,
        "last_update_epoch": None,
        "counters": {"runs_total": 0, "runs_success": 0, "runs_error": 0},
        "run": {
            "id": None,
            "status": None,
            "started_at": None,
            "started_at_epoch": None,
            "last_stage": None,
            "last_stage_started_at": None,
            "last_stage_started_at_epoch": None,
        },
        "stages": {},
        "last": {},
        "errors": [],
    }


@dataclass
class Monitor:
    repo: str
    status_dir: Path
    enabled: bool = True

    def _status_path(self) -> Path:
        return self.status_dir / f"{self.repo}.json"

    def _load_state(self) -> Dict[str, Any]:
        path = self._status_path()
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return _default_state(self.repo)

    def _save_state(self, state: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.status_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._status_path().with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._status_path())

    def _update(self, fn) -> None:
        if not self.enabled:
            return
        state = self._load_state()
        fn(state)
        epoch, iso = _now()
        state["last_update"] = iso
        state["last_update_epoch"] = epoch
        self._save_state(state)

    def run_start(self, run_id: Optional[str] = None, topic: Optional[str] = None,
                  script_path: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> None:
        def _apply(state: Dict[str, Any]) -> None:
            state["counters"]["runs_total"] = state.get("counters", {}).get("runs_total", 0) + 1
            epoch, iso = _now()
            state["run"].update({
                "id": run_id,
                "status": "running",
                "started_at": iso,
                "started_at_epoch": epoch,
                "last_stage": None,
                "last_stage_started_at": None,
                "last_stage_started_at_epoch": None,
            })
            if topic:
                state.setdefault("last", {})["topic"] = topic
            if script_path:
                state.setdefault("last", {})["script_path"] = script_path
            if meta:
                state.setdefault("last", {}).update(meta)
        self._update(_apply)

    def stage_start(self, stage: str, meta: Optional[Dict[str, Any]] = None) -> None:
        def _apply(state: Dict[str, Any]) -> None:
            epoch, iso = _now()
            state["run"]["last_stage"] = stage
            state["run"]["last_stage_started_at"] = iso
            state["run"]["last_stage_started_at_epoch"] = epoch
            stages = state.setdefault("stages", {})
            entry = stages.setdefault(stage, {})
            entry["last_start"] = iso
            entry["last_start_epoch"] = epoch
            if meta:
                entry.setdefault("meta", {}).update(meta)
        self._update(_apply)

    def stage_end(self, stage: str, ok: bool = True, error: Optional[str] = None,
                  meta: Optional[Dict[str, Any]] = None) -> None:
        def _apply(state: Dict[str, Any]) -> None:
            epoch, iso = _now()
            stages = state.setdefault("stages", {})
            entry = stages.setdefault(stage, {})
            entry["last_end"] = iso
            entry["last_end_epoch"] = epoch
            start_epoch = entry.get("last_start_epoch") or state.get("run", {}).get("last_stage_started_at_epoch")
            if start_epoch:
                entry["last_elapsed_sec"] = max(0.0, epoch - start_epoch)
            entry["last_status"] = "success" if ok else "error"
            if error:
                entry["last_error"] = error
                state.setdefault("errors", []).append({
                    "stage": stage,
                    "error": error,
                    "ts": iso,
                })
            if meta:
                entry.setdefault("meta", {}).update(meta)
        self._update(_apply)

    def run_success(self, final_video: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> None:
        def _apply(state: Dict[str, Any]) -> None:
            state["counters"]["runs_success"] = state.get("counters", {}).get("runs_success", 0) + 1
            state["run"]["status"] = "success"
            if final_video:
                state.setdefault("last", {})["final_video"] = final_video
            if meta:
                state.setdefault("last", {}).update(meta)
        self._update(_apply)

    def run_error(self, error: str) -> None:
        def _apply(state: Dict[str, Any]) -> None:
            state["counters"]["runs_error"] = state.get("counters", {}).get("runs_error", 0) + 1
            state["run"]["status"] = "error"
            state.setdefault("last", {})["error"] = error
            state.setdefault("errors", []).append({"stage": state.get("run", {}).get("last_stage"), "error": error})
        self._update(_apply)

    def set_last_meta(self, meta: Dict[str, Any]) -> None:
        def _apply(state: Dict[str, Any]) -> None:
            if meta:
                state.setdefault("last", {}).update(meta)
        self._update(_apply)


def get_monitor(repo: str, base_dir: Optional[Path] = None) -> Monitor:
    enable_env = os.getenv("MONITOR_ENABLE", "1").strip().lower()
    enabled = enable_env in ("1", "true", "yes", "on")

    if base_dir is None:
        base_dir = Path.cwd()
    base_dir = Path(base_dir).resolve()

    root = base_dir.parent if (base_dir.parent / "monitor").exists() else base_dir
    status_dir = Path(os.getenv("MONITOR_STATUS_DIR", str(root / "monitor" / "status"))).resolve()

    return Monitor(repo=repo, status_dir=status_dir, enabled=enabled)


def run_health_server(repo: str, base_dir: Optional[Path] = None,
                      host: str = "0.0.0.0", port: Optional[int] = None) -> None:
    monitor = get_monitor(repo, base_dir=base_dir)
    status_path = monitor.status_dir / f"{repo}.json"
    port_env = os.getenv("MONITOR_HEALTH_PORT")
    if port is None:
        port = int(port_env) if port_env else 9100
    base_dir = Path(base_dir or Path.cwd()).resolve()
    comment_state = base_dir / "fb_user_thread_state.json"
    comment_config = base_dir / "fb_llm_config.json"
    scheduler_status = base_dir / "scheduler_status.json"

    def _read_comment_state() -> Dict[str, Any]:
        if not comment_state.exists():
            return {"enabled": False}
        try:
            state = json.loads(comment_state.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"enabled": True, "error": f"state_read_failed: {exc}"}

        post_count = len(state) if isinstance(state, dict) else 0
        user_count = 0
        last_reply = None
        if isinstance(state, dict):
            for post_map in state.values():
                if isinstance(post_map, dict):
                    user_count += len(post_map)
                    for entry in post_map.values():
                        if isinstance(entry, dict):
                            ts = entry.get("last_page_reply_time")
                            if ts and (last_reply is None or ts > last_reply):
                                last_reply = ts
        config_info = {}
        if comment_config.exists():
            try:
                cfg = json.loads(comment_config.read_text(encoding="utf-8"))
                config_info = {
                    "dry_run": cfg.get("dry_run"),
                    "max_posts": cfg.get("max_posts"),
                    "lookback_hours": cfg.get("lookback_hours"),
                    "max_comments_per_post": cfg.get("max_comments_per_post"),
                }
            except Exception as exc:
                config_info = {"error": f"config_read_failed: {exc}"}

        return {
            "enabled": True,
            "state_file": str(comment_state),
            "config_file": str(comment_config) if comment_config.exists() else None,
            "posts": post_count,
            "users": user_count,
            "last_page_reply_time": last_reply,
            **config_info,
        }

    def _read_scheduler_state() -> Dict[str, Any]:
        if not scheduler_status.exists():
            return {"enabled": False}
        try:
            data = json.loads(scheduler_status.read_text(encoding="utf-8"))
            return {"enabled": True, "status": data}
        except Exception as exc:
            return {"enabled": True, "error": f"scheduler_read_failed: {exc}"}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: Dict[str, Any]) -> None:
            payload = json.dumps(body, indent=2, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):  # noqa: N802
            if self.path in ("/health", "/healthz"):
                if status_path.exists():
                    state = monitor._load_state()
                    self._send(200, {
                        "status": "ok",
                        "repo": repo,
                        "last_update": state.get("last_update"),
                        "comment": _read_comment_state(),
                        "scheduler": _read_scheduler_state(),
                    })
                else:
                    self._send(503, {
                        "status": "missing",
                        "repo": repo,
                        "comment": _read_comment_state(),
                        "scheduler": _read_scheduler_state(),
                    })
                return
            if self.path in ("/status", "/status.json"):
                if status_path.exists():
                    state = monitor._load_state()
                    state["comment"] = _read_comment_state()
                    state["scheduler"] = _read_scheduler_state()
                    self._send(200, state)
                else:
                    self._send(404, {
                        "error": "status not found",
                        "repo": repo,
                        "comment": _read_comment_state(),
                        "scheduler": _read_scheduler_state(),
                    })
                return
            self._send(404, {"error": "not found"})

        def log_message(self, format, *args):  # noqa: A003
            return

    server = HTTPServer((host, port), Handler)
    print(f"[monitor] {repo} health server listening on {host}:{port} status={status_path}")
    server.serve_forever()
