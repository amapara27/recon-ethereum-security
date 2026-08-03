import os
import sqlite3
import time
import joblib

from pathlib import Path
from dotenv import load_dotenv
from web3 import Web3

from feature_pipeline import get_feature_vector, load_master_column_list

# Loading files
base_dir = Path(__file__).parent
lists_dir = base_dir.parent / "config/lists"
models_dir = base_dir.parent / "models"
env_path = base_dir.parent.parent / ".env"
alerts_path = base_dir.parent.parent / "database/alerts.db"

# Ensure the directory for alerts.db exists
alerts_path.parent.mkdir(parents=True, exist_ok=True)

load_dotenv(env_path)
rpc_url = os.environ.get('ALCHEMY_RPC_URL')

POLL_SECONDS = int(os.environ.get('POLL_SECONDS', 12))       # mainnet block time
MAX_ADDR_PER_BLOCK = int(os.environ.get('MAX_ADDR_PER_BLOCK', 5))  # each addr = 2 Etherscan calls

MODEL_PATH = models_dir / "fraud_model.joblib"
MASTER_COLUMN_PATH = lists_dir / "master_column_list.txt"
MASTER_SENT_PATH = lists_dir / "master_sent.txt"
MASTER_REC_PATH = lists_dir / "master_rec.txt"
MASTER_COLUMN_LIST = load_master_column_list(MASTER_COLUMN_PATH)

model = joblib.load(MODEL_PATH)
w3 = Web3(Web3.HTTPProvider(rpc_url))

processed_addr = set()

def setup_databse():
    conn = sqlite3.connect(alerts_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            "timestamp" DATETIME DEFAULT CURRENT_TIMESTAMP,
            "address" TEXT NOT NULL,
            "tx_hash" TEXT NOT NULL UNIQUE,
            "probability" DOUBLE NOT NULL
        )
    ''')

    # Add missing columns if they don't exist
    cursor.execute("PRAGMA table_info(alerts)")
    columns = [info[1] for info in cursor.fetchall()]
    
    if "to_address" not in columns:
        print("Migrating: Adding to_address column")
        cursor.execute('ALTER TABLE alerts ADD COLUMN "to_address" TEXT')
    
    if "value" not in columns:
        print("Migrating: Adding value column")
        cursor.execute('ALTER TABLE alerts ADD COLUMN "value" TEXT')

    conn.commit()
    conn.close()

def log_transaction(address, to_address, value, tx_hash, probability):
    conn = sqlite3.connect(alerts_path)
    cursor = conn.cursor()

    data = (address, to_address, value, tx_hash, probability)
    sql = "INSERT OR IGNORE INTO alerts (address, to_address, value, tx_hash, probability) VALUES (?, ?, ?, ?, ?)"

    cursor.execute(sql, data)

    conn.commit()
    conn.close()

def clean_database():
    conn = sqlite3.connect(alerts_path)
    cursor = conn.cursor()

    try:
        cursor.execute('''
            DELETE FROM alerts WHERE timestamp < datetime('now', '-1 day')
        ''')

    except Exception as e:
        print(f"Cleanup Error: {e}")

    conn.commit()
    conn.close()

def pick_addresses(block):
    """First unseen sender per address in the block, capped at MAX_ADDR_PER_BLOCK."""
    fresh = {}

    for tx in block.transactions:
        if tx['from'] not in processed_addr and tx['from'] not in fresh:
            fresh[tx['from']] = tx

            if len(fresh) >= MAX_ADDR_PER_BLOCK:
                break

    return fresh

def process_block(block):
    fresh = pick_addresses(block)
    print(f"📦 Block {block['number']}: {len(block.transactions)} txs, scoring {len(fresh)}")

    for from_addr, tx in fresh.items():
        try:
            final_vector = get_feature_vector(from_addr, MASTER_SENT_PATH, MASTER_REC_PATH, MASTER_COLUMN_LIST)

            if final_vector.empty:
                continue

            fraud_probability = model.predict_proba(final_vector)[0][1]

            value_eth = w3.from_wei(tx['value'], 'ether')
            log_transaction(from_addr, tx['to'], str(value_eth), tx['hash'].hex(), fraud_probability)

            if fraud_probability >= 0.3:
                print(f"🚨 FRAUD ALERT! {from_addr} (Probability: {fraud_probability})")

            processed_addr.add(from_addr)

            if len(processed_addr) > 10000:
                processed_addr.clear()

        except Exception as e:
            print(f"Error scoring {from_addr}: {e}")

def main_loop():
    if not w3.is_connected():
        print("Connection failed. Check your RPC_URL.")
        return

    last_clean = time.time()
    last_block = None

    while True:
        if time.time() - last_clean > 3600:
            clean_database()
            last_clean = time.time()

        try:
            # ponytail: poll 'latest' instead of a block filter — filters expire on Alchemy, and
            # get_new_entries() returns the whole backlog. Skipped blocks are dropped on purpose.
            block = w3.eth.get_block('latest', full_transactions=True)

            if block['number'] != last_block:
                last_block = block['number']
                process_block(block)

        except Exception as e:
            print(f"Loop error: {e}")

        time.sleep(POLL_SECONDS)

def main():
    setup_databse()
    main_loop()

if __name__ == "__main__":
    main()
