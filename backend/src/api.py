import sqlite3
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
from typing import List
from pydantic import BaseModel

base_dir = Path(__file__).parent
alerts_path = base_dir.parent / "alerts.db"

app = FastAPI()

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
def get_latest_alerts():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        entries = cursor.execute('''
            SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 50
            ''')

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

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
