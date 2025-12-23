import sqlite3
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
from typing import List
from pydantic import BaseModel

from contract_fetcher import fetch_source_code
from contract_analyzer import analyze_smart_contract

base_dir = Path(__file__).parent
alerts_path = base_dir.parent / "alerts.db"

app = FastAPI()

class ContractRequest(BaseModel):
    address: str

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
            alert = {
                'id': entry['id'],
                'address': entry['address'],
                'timestamp': entry['timestamp'],
                'tx_hash': entry['tx_hash'],
                'probability': entry['probability']
            }

            alerts.append(alert)

        return alerts

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        conn.close()

@app.post("/api/contract-analyzer")
async def analyze_contract(request : ContractRequest):
    target_address = request.address

    source_code = fetch_source_code(target_address)

    if source_code == "" or "Unknown" in source_code:
        raise HTTPException(status_code=404, detail="Contract Source Code Not Found")

    return analyze_smart_contract(source_code)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
