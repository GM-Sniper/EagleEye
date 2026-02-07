
import sys
from pathlib import Path
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import time

import httpx

BASE = "http://127.0.0.1:8000"


def wait_for_health(timeout_seconds: int = 240) -> None:
    deadline = time.time() + timeout_seconds
    last = None

    while time.time() < deadline:
        try:
            r = httpx.get(f"{BASE}/health", timeout=10)
            r.raise_for_status()
            payload = r.json()
            last = payload
            if payload.get("status") == "healthy":
                print("API healthy")
                return
            print("waiting...", payload.get("status"))
        except Exception as exc:
            print("waiting...", str(exc)[:160])
        time.sleep(2)

    raise SystemExit(f"API did not become healthy in time; last={last}")


def main() -> int:
    wait_for_health()

    print("GET /forecast")
    r = httpx.get(f"{BASE}/forecast", params={"days": 3}, timeout=60)
    print(r.status_code)
    payload = r.json()
    print("keys:", sorted(payload.keys()))
    print("forecast_len:", len(payload.get("forecast", [])))

    print("GET /inventory/recommendations")
    r = httpx.get(f"{BASE}/inventory/recommendations", params={"top_n": 5}, timeout=60)
    print(r.status_code)
    payload = r.json()
    print("keys:", sorted(payload.keys()))
    items = payload.get("items", [])
    print("items_len:", len(items))

    if items:
        item_id = items[0].get("item_id")
        print("GET /inventory/item", item_id)
        r = httpx.get(f"{BASE}/inventory/item/{item_id}", params={"forecast_days": 7}, timeout=120)
        print(r.status_code)
        item_payload = r.json()
        print("item keys:", sorted(item_payload.keys()))
        print("forecast_len:", len(item_payload.get("forecast", [])))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())