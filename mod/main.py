import comfy.options
comfy.options.enable_args_parsing()

import os
import importlib.util
import folder_paths
import time
from comfy.cli_args import args
from app.logger import setup_logger
from app.assets.scanner import seed_assets
import itertools
import utils.extra_config
import logging
import sys
from comfy_execution.progress import get_progress_state
from comfy_execution.utils import get_executing_context
from comfy_api import feature_flags

# =========================
# NEW: server-side /prompt logging + GUI/API normalization
# =========================
import json
from pathlib import Path
from starlette.requests import Request
from starlette.responses import JSONResponse
from logging.handlers import RotatingFileHandler

_PROMPT_ENQUEUE_TS = {}
_COMFY_LOGGER = None

def _get_comfy_logger():
    global _COMFY_LOGGER
    if _COMFY_LOGGER:
        return _COMFY_LOGGER
    logger = logging.getLogger("comfy_api_full")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        log_dir = Path(os.getenv("COMFY_LOG_DIR", "./logs")).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            log_dir / "comfy_api_full.log",
            maxBytes=20 * 1024 * 1024,
            backupCount=5,
        )
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    _COMFY_LOGGER = logger
    return logger

def _log_comfy(event: str, payload: dict | None = None):
    logger = _get_comfy_logger()
    if payload is None:
        logger.info(event)
        return
    try:
        line = json.dumps({"event": event, **payload}, ensure_ascii=False)
    except Exception:
        line = f"{event} {payload}"
    logger.info(line)


def _install_prompt_logging_and_gui_compat(prompt_server):
    """
    Goal:
      Make API submissions execute the SAME way the GUI does by ensuring the server
      always sees a GUI-shaped payload for /prompt:
          {"prompt": <workflow_dict>, "client_id": "...", "extra_data": {...}}

      Also logs exactly what the server received so you can diff GUI vs API.

    Backwards compatibility:
      - If caller posts raw workflow dict (legacy), we wrap it into GUI shape.
      - If caller already posts GUI shape, we pass it through unchanged.
      - Logging can be toggled via env without changing behavior.

    Env:
      COMFY_LOG_PROMPTS=1
      COMFY_LOG_DIR=./logs
      COMFY_LOG_PRETTY=1
      COMFY_FORCE_GUI_SHAPE=1   # always normalize into GUI shape
    """

    ENABLE_LOG = os.getenv("COMFY_LOG_PROMPTS", "0").lower() in ("1", "true", "yes")
    LOG_DIR = Path(os.getenv("COMFY_LOG_DIR", "./logs")).resolve()
    PRETTY = os.getenv("COMFY_LOG_PRETTY", "1").lower() in ("1", "true", "yes")
    FORCE_GUI_SHAPE = os.getenv("COMFY_FORCE_GUI_SHAPE", "1").lower() in ("1", "true", "yes")

    def _safe_write(path: Path, obj: dict):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                if PRETTY:
                    json.dump(obj, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(obj, f, separators=(",", ":"), ensure_ascii=False)
            tmp.replace(path)
        except Exception:
            # never break execution if logging fails
            logging.exception("[COMFY] failed writing prompt log")

    def _normalize_to_gui_shape(payload: object) -> dict:
        """
        Normalize request JSON to the GUI-style /prompt shape.

        GUI typically sends:
          {"prompt": {...}, "client_id": "...", "extra_data": {...}}
        Some clients send:
          {"prompt": {...}, "clientId": "..."}
        Legacy clients send:
          {...workflow...}

        We normalize all of those into:
          {"prompt": workflow, "client_id": <string>, "extra_data": <dict>}
        """
        if not isinstance(payload, dict):
            raise ValueError("Body must be a JSON object")

        # Already GUI shape
        if "prompt" in payload and isinstance(payload.get("prompt"), dict):
            out = dict(payload)

            # normalize key name
            if "client_id" not in out and "clientId" in out:
                out["client_id"] = out.get("clientId")
            if "client_id" not in out:
                out["client_id"] = out.get("client_id") or out.get("clientId") or "api"

            # ensure extra_data exists (many parts of the stack assume it can exist)
            if "extra_data" not in out or not isinstance(out.get("extra_data"), dict):
                out["extra_data"] = {}

            # add a marker so you can distinguish sources in logs
            out["extra_data"].setdefault("_submitted_via", "api" if out["client_id"] == "api" else "client")
            return out

        # Legacy: treat dict as prompt graph/workflow
        client_id = None
        if isinstance(payload.get("client_id"), str):
            client_id = payload.get("client_id")
        elif isinstance(payload.get("clientId"), str):
            client_id = payload.get("clientId")
        if not client_id:
            client_id = "legacy"

        return {
            "prompt": payload,
            "client_id": client_id,
            "extra_data": {"_submitted_via": "legacy"},
        }

    # PromptServer should expose the underlying Starlette app at prompt_server.app
    app = getattr(prompt_server, "app", None)
    if app is None:
        logging.warning("[COMFY] Cannot install /prompt wrapper: prompt_server.app missing")
        return

    # Locate /prompt route and wrap it
    target_route = None
    for r in getattr(app, "routes", []):
        if getattr(r, "path", None) == "/prompt":
            target_route = r
            break

    if target_route is None:
        logging.warning("[COMFY] Cannot install /prompt wrapper: /prompt route not found")
        return

    orig_endpoint = target_route.endpoint

    async def wrapped_endpoint(request: Request):
        t0 = time.perf_counter()
        req_id = f"{int(time.time())}-{os.getpid()}"

        raw = b""
        received_json = None
        normalized_json = None

        try:
            raw = await request.body()  # cached by Starlette; safe to read here
            try:
                received_json = json.loads(raw.decode("utf-8", "ignore"))
            except Exception as e:
                return JSONResponse({"error": f"invalid JSON body: {type(e).__name__}: {e}"}, status_code=400)

            if FORCE_GUI_SHAPE:
                try:
                    normalized_json = _normalize_to_gui_shape(received_json)
                except Exception as e:
                    return JSONResponse({"error": f"invalid /prompt payload: {type(e).__name__}: {e}"}, status_code=400)

                # IMPORTANT: rewrite body to normalized GUI shape so the existing /prompt handler
                # sees the same structure as it would from the browser GUI.
                request._body = json.dumps(normalized_json).encode("utf-8")

            if ENABLE_LOG:
                _safe_write(LOG_DIR / "comfy_last_prompt_received.json", {
                    "req_id": req_id,
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "remote": request.client.host if request.client else None,
                    "headers": {
                        "content_type": request.headers.get("content-type"),
                        "content_length": request.headers.get("content-length"),
                        "user_agent": request.headers.get("user-agent"),
                    },
                    "raw_bytes": len(raw),
                    "json_received": received_json,
                    "json_normalized": normalized_json,
                })
            _log_comfy("prompt_received", {
                "req_id": req_id,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "raw_bytes": len(raw),
                "client": request.client.host if request.client else None,
                "force_gui_shape": FORCE_GUI_SHAPE,
            })
            if received_json is not None:
                _log_comfy("prompt_received_json", {"req_id": req_id, "json": received_json})
            if normalized_json is not None:
                _log_comfy("prompt_normalized_json", {"req_id": req_id, "json": normalized_json})

            resp = await orig_endpoint(request)

            dt = time.perf_counter() - t0
            if ENABLE_LOG:
                _safe_write(LOG_DIR / "comfy_last_prompt_response.json", {
                    "req_id": req_id,
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_ms": int(dt * 1000),
                    "status": getattr(resp, "status_code", None),
                })
            prompt_id = None
            try:
                body = getattr(resp, "body", None)
                if body:
                    data = json.loads(body.decode("utf-8", "ignore"))
                    prompt_id = data.get("prompt_id")
            except Exception:
                prompt_id = None
            if prompt_id:
                _PROMPT_ENQUEUE_TS[prompt_id] = time.perf_counter()
            _log_comfy("prompt_response", {
                "req_id": req_id,
                "status": getattr(resp, "status_code", None),
                "elapsed_ms": int(dt * 1000),
                "prompt_id": prompt_id,
            })

            logging.info(
                "[COMFY] /prompt req_id=%s bytes=%d status=%s took=%.3fs",
                req_id, len(raw), getattr(resp, "status_code", None), dt
            )
            return resp

        except Exception as e:
            # Never block execution; fall back to original handler if wrapper fails
            logging.exception("[COMFY] /prompt wrapper error (falling back): %s", e)
            return await orig_endpoint(request)

    target_route.endpoint = wrapped_endpoint
    logging.info(
        "[COMFY] Installed /prompt GUI-compat wrapper. COMFY_LOG_PROMPTS=%s COMFY_FORCE_GUI_SHAPE=%s LOG_DIR=%s",
        ENABLE_LOG, FORCE_GUI_SHAPE, str(LOG_DIR)
    )

    # Locate /history route and wrap it (full history logging)
    history_route = None
    for r in getattr(app, "routes", []):
        if getattr(r, "path", None) in ("/history/{prompt_id}", "/history/{prompt_id:path}"):
            history_route = r
            break

    if history_route is None:
        logging.warning("[COMFY] Cannot install /history wrapper: /history route not found")
        return

    history_endpoint = history_route.endpoint

    async def wrapped_history(request: Request):
        t0 = time.perf_counter()
        prompt_id = None
        try:
            prompt_id = request.path_params.get("prompt_id")
        except Exception:
            prompt_id = None

        resp = await history_endpoint(request)
        dt = time.perf_counter() - t0

        try:
            body = getattr(resp, "body", None)
            if body:
                data = json.loads(body.decode("utf-8", "ignore"))
                _log_comfy("history_response", {
                    "prompt_id": prompt_id,
                    "elapsed_ms": int(dt * 1000),
                    "json": data,
                })
            else:
                _log_comfy("history_response", {
                    "prompt_id": prompt_id,
                    "elapsed_ms": int(dt * 1000),
                    "json": None,
                })
        except Exception as e:
            _log_comfy("history_response_error", {
                "prompt_id": prompt_id,
                "elapsed_ms": int(dt * 1000),
                "error": f"{type(e).__name__}: {e}",
            })
        return resp

    history_route.endpoint = wrapped_history
    logging.info("[COMFY] Installed /history wrapper for full history logging.")


# ---------------------------
# NOTE: These do not do anything on core ComfyUI, they are for custom nodes.
# ---------------------------
if __name__ == "__main__":
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'

setup_logger(log_level=args.verbose, use_stdout=args.log_stdout)

if os.name == "nt":
    os.environ['MIMALLOC_PURGE_DELAY'] = '0'

if __name__ == "__main__":
    os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'
    if args.default_device is not None:
        default_dev = args.default_device
        devices = list(range(32))
        devices.remove(default_dev)
        devices.insert(0, default_dev)
        devices = ','.join(map(str, devices))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(devices)
        os.environ['HIP_VISIBLE_DEVICES'] = str(devices)

    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        os.environ['HIP_VISIBLE_DEVICES'] = str(args.cuda_device)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.cuda_device)
        logging.info("Set cuda device to: {}".format(args.cuda_device))

    if args.oneapi_device_selector is not None:
        os.environ['ONEAPI_DEVICE_SELECTOR'] = args.oneapi_device_selector
        logging.info("Set oneapi device selector to: {}".format(args.oneapi_device_selector))

    if args.deterministic:
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    import cuda_malloc
    if "rocm" in cuda_malloc.get_torch_version_noimport():
        os.environ['OCL_SET_SVM_SIZE'] = '262144'  # set at the request of AMD


def handle_comfyui_manager_unavailable():
    if not args.windows_standalone_build:
        logging.warning(
            f"\n\nYou appear to be running comfyui-manager from source, this is not recommended. "
            f"Please install comfyui-manager using the following command:\ncommand:\n\t{sys.executable} -m pip install --pre comfyui_manager\n"
        )
    args.enable_manager = False


if args.enable_manager:
    if importlib.util.find_spec("comfyui_manager"):
        import comfyui_manager

        if not comfyui_manager.__file__ or not comfyui_manager.__file__.endswith('__init__.py'):
            handle_comfyui_manager_unavailable()
    else:
        handle_comfyui_manager_unavailable()


def apply_custom_paths():
    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        utils.extra_config.load_extra_path_config(extra_model_paths_config_path)

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            utils.extra_config.load_extra_path_config(config_path)

    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        logging.info(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))
    folder_paths.add_model_folder_path("diffusion_models", os.path.join(folder_paths.get_output_directory(), "diffusion_models"))
    folder_paths.add_model_folder_path("loras", os.path.join(folder_paths.get_output_directory(), "loras"))

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        logging.info(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    if args.user_directory:
        user_dir = os.path.abspath(args.user_directory)
        logging.info(f"Setting user directory to: {user_dir}")
        folder_paths.set_user_directory(user_dir)


def execute_prestartup_script():
    if args.disable_all_custom_nodes and len(args.whitelist_custom_nodes) == 0:
        return

    def execute_script(script_path):
        module_name = os.path.splitext(script_path)[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True
        except Exception as e:
            logging.error(f"Failed to execute startup-script: {script_path} / {e}")
        return False

    node_paths = folder_paths.get_folder_paths("custom_nodes")
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)
        node_prestartup_times = []

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)

            if args.enable_manager:
                if comfyui_manager.should_be_disabled(module_path):
                    continue

            if os.path.isfile(module_path) or module_path.endswith(".disabled") or module_path == "__pycache__":
                continue

            script_path = os.path.join(module_path, "prestartup_script.py")
            if os.path.exists(script_path):
                if args.disable_all_custom_nodes and possible_module not in args.whitelist_custom_nodes:
                    logging.info(f"Prestartup Skipping {possible_module} due to disable_all_custom_nodes and whitelist_custom_nodes")
                    continue
                time_before = time.perf_counter()
                success = execute_script(script_path)
                node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))

    if len(node_prestartup_times) > 0:
        logging.info("\nPrestartup times for custom nodes:")
        for n in sorted(node_prestartup_times):
            import_message = "" if n[2] else " (PRESTARTUP FAILED)"
            logging.info("{:6.1f} seconds{}: {}".format(n[0], import_message, n[1]))
        logging.info("")


apply_custom_paths()

if args.enable_manager:
    comfyui_manager.prestartup()

execute_prestartup_script()

# Main code
import asyncio
import shutil
import threading
import gc

if 'torch' in sys.modules:
    logging.warning("WARNING: Potential Error in code: Torch already imported, torch should never be imported before this point.")

import comfy.utils

import execution
import server
from protocol import BinaryEventTypes
import nodes
import comfy.model_management
import comfyui_version
import app.logger
import hook_breaker_ac10a0


def cuda_malloc_warning():
    device = comfy.model_management.get_torch_device()
    device_name = comfy.model_management.get_torch_device_name(device)
    cuda_malloc_warning = False
    if "cudaMallocAsync" in device_name:
        for b in cuda_malloc.blacklist:
            if b in device_name:
                cuda_malloc_warning = True
        if cuda_malloc_warning:
            logging.warning(
                "\nWARNING: this card most likely does not support cuda-malloc, "
                "if you get \"CUDA error\" please run ComfyUI with: --disable-cuda-malloc\n"
            )


def prompt_worker(q, server_instance):
    current_time: float = 0.0
    cache_type = execution.CacheType.CLASSIC
    if args.cache_lru > 0:
        cache_type = execution.CacheType.LRU
    elif args.cache_ram > 0:
        cache_type = execution.CacheType.RAM_PRESSURE
    elif args.cache_none:
        cache_type = execution.CacheType.NONE

    e = execution.PromptExecutor(
        server_instance,
        cache_type=cache_type,
        cache_args={"lru": args.cache_lru, "ram": args.cache_ram}
    )
    last_gc_collect = 0
    need_gc = False
    gc_collect_interval = 10.0

    while True:
        timeout = 1000.0
        if need_gc:
            timeout = max(gc_collect_interval - (current_time - last_gc_collect), 0.0)

        queue_item = q.get(timeout=timeout)
        if queue_item is not None:
            item, item_id = queue_item
            execution_start_time = time.perf_counter()
            prompt_id = item[1]
            server_instance.last_prompt_id = prompt_id
            enqueue_ts = _PROMPT_ENQUEUE_TS.pop(prompt_id, None)
            if enqueue_ts is not None:
                queue_wait = execution_start_time - enqueue_ts
                _log_comfy("prompt_queue_wait", {
                    "prompt_id": prompt_id,
                    "queue_wait_s": round(queue_wait, 3),
                })

            sensitive = item[5]
            extra_data = item[3].copy()
            for k in sensitive:
                extra_data[k] = sensitive[k]

            e.execute(item[2], prompt_id, extra_data, item[4])
            need_gc = True

            remove_sensitive = lambda prompt: prompt[:5] + prompt[6:]
            q.task_done(
                item_id,
                e.history_result,
                status=execution.PromptQueue.ExecutionStatus(
                    status_str='success' if e.success else 'error',
                    completed=e.success,
                    messages=e.status_messages
                ),
                process_item=remove_sensitive
            )
            if server_instance.client_id is not None:
                server_instance.send_sync("executing", {"node": None, "prompt_id": prompt_id}, server_instance.client_id)

            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time
            _log_comfy("prompt_execution_done", {
                "prompt_id": prompt_id,
                "execution_time_s": round(execution_time, 3),
                "status": "success" if e.success else "error",
            })

            if execution_time > 600:
                execution_time = time.strftime("%H:%M:%S", time.gmtime(execution_time))
                logging.info(f"Prompt executed in {execution_time}")
            else:
                logging.info("Prompt executed in {:.2f} seconds".format(execution_time))

        flags = q.get_flags()
        free_memory = flags.get("free_memory", False)

        if flags.get("unload_models", free_memory):
            comfy.model_management.unload_all_models()
            need_gc = True
            last_gc_collect = 0

        if free_memory:
            e.reset()
            need_gc = True
            last_gc_collect = 0

        if need_gc:
            current_time = time.perf_counter()
            if (current_time - last_gc_collect) > gc_collect_interval:
                gc.collect()
                comfy.model_management.soft_empty_cache()
                last_gc_collect = current_time
                need_gc = False
                hook_breaker_ac10a0.restore_functions()


async def run(server_instance, address='', port=8188, verbose=True, call_on_start=None):
    addresses = []
    for addr in address.split(","):
        addresses.append((addr, port))
    await asyncio.gather(
        server_instance.start_multi_address(addresses, call_on_start, verbose),
        server_instance.publish_loop()
    )


def hijack_progress(server_instance):
    def hook(value, total, preview_image, prompt_id=None, node_id=None):
        executing_context = get_executing_context()
        if prompt_id is None and executing_context is not None:
            prompt_id = executing_context.prompt_id
        if node_id is None and executing_context is not None:
            node_id = executing_context.node_id
        comfy.model_management.throw_exception_if_processing_interrupted()
        if prompt_id is None:
            prompt_id = server_instance.last_prompt_id
        if node_id is None:
            node_id = server_instance.last_node_id
        progress = {"value": value, "max": total, "prompt_id": prompt_id, "node": node_id}
        get_progress_state().update_progress(node_id, value, total, preview_image)

        server_instance.send_sync("progress", progress, server_instance.client_id)
        if preview_image is not None:
            if not feature_flags.supports_feature(
                server_instance.sockets_metadata,
                server_instance.client_id,
                "supports_preview_metadata",
            ):
                server_instance.send_sync(
                    BinaryEventTypes.UNENCODED_PREVIEW_IMAGE,
                    preview_image,
                    server_instance.client_id,
                )

    comfy.utils.set_progress_bar_global_hook(hook)


def cleanup_temp():
    temp_dir = folder_paths.get_temp_directory()
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


def setup_database():
    try:
        from app.database.db import init_db, dependencies_available
        if dependencies_available():
            init_db()
            if not args.disable_assets_autoscan:
                seed_assets(["models"], enable_logging=True)
    except Exception as e:
        logging.error(
            "Failed to initialize database. Please ensure you have installed the latest requirements. "
            f"If the error persists, please report this as in future the database will be required: {e}"
        )


def start_comfyui(asyncio_loop=None):
    """
    Starts the ComfyUI server using the provided asyncio event loop or creates a new one.
    Returns the event loop, server instance, and a function to start the server asynchronously.
    """
    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        logging.info(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()

    if args.windows_standalone_build:
        try:
            import new_updater
            new_updater.update_windows_updater()
        except:
            pass

    if not asyncio_loop:
        asyncio_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(asyncio_loop)

    prompt_server = server.PromptServer(asyncio_loop)

    if args.enable_manager and not args.disable_manager_ui:
        comfyui_manager.start()

    hook_breaker_ac10a0.save_functions()
    asyncio_loop.run_until_complete(nodes.init_extra_nodes(
        init_custom_nodes=(not args.disable_all_custom_nodes) or len(args.whitelist_custom_nodes) > 0,
        init_api_nodes=not args.disable_api_nodes
    ))
    hook_breaker_ac10a0.restore_functions()

    cuda_malloc_warning()
    setup_database()

    prompt_server.add_routes()

    # =========================
    # NEW: ensure API /prompt shape matches GUI
    # =========================
    _install_prompt_logging_and_gui_compat(prompt_server)

    hijack_progress(prompt_server)

    threading.Thread(
        target=prompt_worker,
        daemon=True,
        args=(prompt_server.prompt_queue, prompt_server,)
    ).start()

    if args.quick_test_for_ci:
        exit(0)

    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
    call_on_start = None
    if args.auto_launch:
        def startup_server(scheme, address, port):
            import webbrowser
            if os.name == 'nt' and address == '0.0.0.0':
                address = '127.0.0.1'
            if ':' in address:
                address = "[{}]".format(address)
            webbrowser.open(f"{scheme}://{address}:{port}")
        call_on_start = startup_server

    async def start_all():
        await prompt_server.setup()
        await run(
            prompt_server,
            address=args.listen,
            port=args.port,
            verbose=not args.dont_print_server,
            call_on_start=call_on_start
        )

    return asyncio_loop, prompt_server, start_all


if __name__ == "__main__":
    logging.info("Python version: {}".format(sys.version))
    logging.info("ComfyUI version: {}".format(comfyui_version.__version__))

    if sys.version_info.major == 3 and sys.version_info.minor < 10:
        logging.warning("WARNING: You are using a python version older than 3.10, please upgrade to a newer one. 3.12 and above is recommended.")

    event_loop, _, start_all_func = start_comfyui()
    try:
        x = start_all_func()
        app.logger.print_startup_warnings()
        event_loop.run_until_complete(x)
    except KeyboardInterrupt:
        logging.info("\nStopped server")

    cleanup_temp()
