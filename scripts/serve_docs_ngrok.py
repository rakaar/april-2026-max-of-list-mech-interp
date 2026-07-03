#!/usr/bin/env python3
"""Serve MkDocs locally and expose it with ngrok when needed."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from pyngrok import conf, ngrok


ROOT = Path(__file__).resolve().parents[1]
HOST = "0.0.0.0"
PORT = 8000


def main() -> int:
    token = os.environ.get("NGROK_TOKEN")
    if not token:
        print("NGROK_TOKEN is not set. Run `source ~/.bashrc` first.", file=sys.stderr)
        return 2

    conf.get_default().auth_token = token

    mkdocs = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "serve",
            "--dev-addr",
            f"{HOST}:{PORT}",
        ],
        cwd=ROOT,
    )

    try:
        time.sleep(3)
        tunnel = ngrok.connect(addr=PORT, bind_tls=True)
        print(f"MkDocs bind: http://{HOST}:{PORT}")
        print(f"ngrok public: {tunnel.public_url}")
        print("Press Ctrl-C to stop.")

        while mkdocs.poll() is None:
            time.sleep(1)
        return mkdocs.returncode or 0
    except KeyboardInterrupt:
        return 130
    finally:
        ngrok.kill()
        if mkdocs.poll() is None:
            mkdocs.send_signal(signal.SIGINT)
            try:
                mkdocs.wait(timeout=10)
            except subprocess.TimeoutExpired:
                mkdocs.kill()


if __name__ == "__main__":
    raise SystemExit(main())
