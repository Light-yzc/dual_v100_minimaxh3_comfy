#!/usr/bin/env python3
"""Ask a local ComfyUI server to unload model/cache state between H3 phases."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Release ComfyUI model and execution caches without stopping the server."
    )
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="Time to let ComfyUI consume the /free queue flag (default: 1 second).",
    )
    args = parser.parse_args()
    if args.settle_seconds < 0:
        raise SystemExit("--settle-seconds must be non-negative")

    url = f"{args.server.rstrip('/')}/free"
    payload = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise SystemExit(f"ComfyUI /free returned HTTP {response.status}")
    if args.settle_seconds:
        time.sleep(args.settle_seconds)
    print("Requested ComfyUI model/cache release.")


if __name__ == "__main__":
    main()
