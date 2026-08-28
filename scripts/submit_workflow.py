#!/usr/bin/env python3
import argparse
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path


def request_json(url: str, payload=None, *, request_timeout: float = 30.0):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=request_timeout) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON file for the submission/final status summary",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=300.0,
        help="per-request HTTP timeout while ComfyUI is busy (default: 300 seconds)",
    )
    args = parser.parse_args()

    def save_summary(summary):
        if args.output is None:
            return
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
    started = time.perf_counter()
    result = request_json(
        f"{args.server.rstrip('/')}/prompt",
        {"prompt": workflow},
        request_timeout=args.request_timeout,
    )
    prompt_id = result["prompt_id"]
    submitted = {"prompt_id": prompt_id, "workflow": str(args.workflow)}
    print(json.dumps(submitted, indent=2))
    if not args.wait:
        save_summary(submitted)
        return

    while True:
        if time.perf_counter() - started > args.timeout:
            raise TimeoutError(f"Prompt {prompt_id} exceeded {args.timeout}s")
        try:
            history = request_json(
                f"{args.server.rstrip('/')}/history/{prompt_id}",
                request_timeout=args.request_timeout,
            )
        except (TimeoutError, socket.timeout):
            # ComfyUI can hold the HTTP event loop during a long model load or
            # denoise.  The total benchmark timeout above remains authoritative.
            continue
        if prompt_id in history:
            elapsed = time.perf_counter() - started
            item = history[prompt_id]
            status = item.get("status", {})
            summary = {
                "prompt_id": prompt_id,
                "workflow": str(args.workflow),
                "elapsed_seconds": round(elapsed, 3),
                "status": status,
                "output_nodes": sorted(item.get("outputs", {}).keys()),
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            save_summary(summary)
            if status.get("status_str") == "error" or status.get("completed") is False:
                raise RuntimeError(
                    f"Prompt {prompt_id} failed: "
                    f"{json.dumps(status, ensure_ascii=False)}"
                )
            return
        time.sleep(2.0)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as exc:
        raise SystemExit(f"ComfyUI request failed: {exc}") from exc
