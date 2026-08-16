#!/usr/bin/env python3
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def request_json(url: str, payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout", type=float, default=7200.0)
    args = parser.parse_args()

    workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
    started = time.perf_counter()
    result = request_json(f"{args.server.rstrip('/')}/prompt", {"prompt": workflow})
    prompt_id = result["prompt_id"]
    print(json.dumps({"prompt_id": prompt_id}, indent=2))
    if not args.wait:
        return

    while True:
        if time.perf_counter() - started > args.timeout:
            raise TimeoutError(f"Prompt {prompt_id} exceeded {args.timeout}s")
        history = request_json(
            f"{args.server.rstrip('/')}/history/{prompt_id}"
        )
        if prompt_id in history:
            elapsed = time.perf_counter() - started
            item = history[prompt_id]
            status = item.get("status", {})
            print(json.dumps({
                "prompt_id": prompt_id,
                "elapsed_seconds": round(elapsed, 3),
                "status": status,
                "output_nodes": sorted(item.get("outputs", {}).keys()),
            }, ensure_ascii=False, indent=2))
            return
        time.sleep(2.0)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as exc:
        raise SystemExit(f"ComfyUI request failed: {exc}") from exc
