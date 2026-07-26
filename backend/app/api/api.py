import re
import sqlite3
import time
import uvicorn
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add backend to sys.path for absolute imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.services.contract_fetcher import fetch_source_code
from app.services.contract_analyzer import analyze_smart_contract, get_cached_analysis

base_dir = Path(__file__).parent
alerts_path = base_dir.parent.parent / "database/alerts.db"
audits_path = base_dir.parent.parent / "database/contract_cache.db"

app = FastAPI()

class ContractRequest(BaseModel):
    address: str

# Global sliding-window rate limit for the paid analyzer endpoint — caps worst-case
# Anthropic/Etherscan spend. Cache handles repeat addresses; this caps distinct ones.
# ponytail: in-memory, single API process. Move to Redis only if you run multiple workers.
ANALYZE_MAX = 3          # max analyzer calls...
ANALYZE_WINDOW = 86400   # ...per rolling 24h, across all callers combined (each slot frees 24h after use)
_analyze_calls = deque()

async def rate_limit_analyzer():
    now = time.monotonic()
    while _analyze_calls and now - _analyze_calls[0] > ANALYZE_WINDOW:
        _analyze_calls.popleft()
    if len(_analyze_calls) >= ANALYZE_MAX:
        retry = int(ANALYZE_WINDOW - (now - _analyze_calls[0])) + 1
        raise HTTPException(status_code=429, detail="Contract analysis usage exceeded, please try again later.",
                            headers={"Retry-After": str(retry)})
    _analyze_calls.append(now)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Alert(BaseModel):
    id: int
    address: str
    to_address: Optional[str] = None
    value: Optional[str] = None
    timestamp: str
    tx_hash: str
    probability: float

def get_db_connection():
    conn = sqlite3.connect(alerts_path)
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/api/get-alerts", response_model=List[Alert])
def get_latest_alerts(limit: int = 200):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        entries = cursor.execute('''
            SELECT * FROM alerts
            WHERE timestamp >= datetime('now', '-24 hours')
            ORDER BY timestamp DESC
            LIMIT ?
            ''', (limit,))

        alerts = []
        for entry in entries:
            db_ts = entry['timestamp']
            if ' ' in db_ts:
                iso_ts = db_ts.replace(' ', 'T') + 'Z'
            else:
                iso_ts = db_ts

            alert = {
                'id': entry['id'],
                'address': entry['address'],
                'to_address': entry['to_address'],
                'value': entry['value'],
                'timestamp': iso_ts,
                'tx_hash': entry['tx_hash'],
                'probability': entry['probability']
            }

            alerts.append(alert)

        return alerts

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()

# Cap source sent to Claude — bounds worst-case per-call token cost
MAX_SOURCE_CHARS = 200_000
ETH_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")

@app.post("/api/contract-analyzer")
async def analyze_contract(request : ContractRequest):
    target_address = request.address.strip()

    # Reject junk before it costs an Etherscan call or a rate-limit slot
    if not ETH_ADDRESS_RE.fullmatch(target_address):
        raise HTTPException(status_code=422, detail="Invalid Ethereum address")

    # Cache hits are free: no rate-limit slot, no Etherscan/Anthropic calls
    cached = get_cached_analysis(target_address)
    if cached:
        return cached

    await rate_limit_analyzer()

    source_code = fetch_source_code(target_address)

    if source_code == "" or "Unknown" in source_code:
        raise HTTPException(status_code=404, detail="Contract Source Code Not Found")

    if len(source_code) > MAX_SOURCE_CHARS:
        raise HTTPException(status_code=413, detail="Contract source too large to analyze")

    # use_cache=False: cache already checked above
    return analyze_smart_contract(source_code, contract_address=target_address, use_cache=False)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
