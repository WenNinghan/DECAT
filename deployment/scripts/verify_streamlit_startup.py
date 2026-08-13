"""Start the bundled Streamlit app briefly and verify its local health endpoint."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    deployment = Path(__file__).resolve().parents[1]
    app = deployment / "streamlit_app.py"
    if not app.is_file():
        raise FileNotFoundError(app)

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.headless",
        "true",
        "--server.port",
        str(args.port),
        "--browser.gatherUsageStats",
        "false",
    ]
    process = subprocess.Popen(
        command,
        cwd=deployment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    health_url = f"http://127.0.0.1:{args.port}/_stcore/health"
    healthy = False
    try:
        for _ in range(20):
            time.sleep(1.5)
            try:
                with urllib.request.urlopen(health_url, timeout=3) as response:
                    if response.status == 200:
                        print(f"Streamlit health check PASSED: {response.read().decode().strip()}")
                        healthy = True
                        break
            except Exception:
                if process.poll() is not None:
                    break
        if not healthy:
            raise RuntimeError("Streamlit health check failed")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

