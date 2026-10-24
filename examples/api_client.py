"""Call a running service (`make serve` or `docker compose up`) and print its metrics.

    python examples/api_client.py http://localhost:8000
"""

from __future__ import annotations

import sys

import httpx

base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
tickets = [
    "I was charged twice for my Team subscription",
    "The headphones arrived broken. Also, please send an invoice.",
    "Is there a way to export everything to Excel? If not, please consider it.",
]
with httpx.Client(base_url=base, timeout=10) as client:
    for text in tickets:
        r = client.post("/classify", json={"text": text}).json()
        print(f"{r['route']:5} {r['intent']:22} {r['priority']:7} reasons={r['reasons']}")
    metrics = client.get("/metrics").text
    print("\n".join(line for line in metrics.splitlines()
                    if line.startswith(("modelroute_requests_total", "modelroute_fallback_ratio"))))
