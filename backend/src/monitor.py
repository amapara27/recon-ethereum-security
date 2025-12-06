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
lists_dir = base_dir / "lists"   
models_dir = base_dir.parent / "models"
env_path = base_dir.parent.parent / ".env" 
alerts_path = base_dir.parent / "alerts.db"

load_dotenv(env_path)
rpc_url = os.environ.get('ALCHEMY_RPC_URL')

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

    conn.commit()
    conn.close()

def log_transaction(address, tx_hash, probability):
    conn = sqlite3.connect(alerts_path)
    cursor = conn.cursor()

    data = (address, tx_hash, probability)
    sql = "INSERT OR IGNORE INTO alerts (address, tx_hash, probability) VALUES (?, ?, ?)"

    cursor.execute(sql, data)

    conn.commit()
    conn.close()

def clean_database():
    conn = sqlite3.connect(alerts_path)
    cursor = conn.connect()

    try:
        cursor.execute('''
            DELETE FROM alerts WHERE timestamp < datetime('now', '-1 day')
        ''')

    except Exception as e:
        print(f"❌ Cleanup Error: {e}")
        
    conn.commit()
    conn.close()

def main_loop():
    if not w3.is_connected():
        print("Connection failed. Check your RPC_URL.")
        return
    
    block_filter = w3.eth.filter('latest')

    last_clean = time.time()

    while True:
        if time.time() - last_clean > 3600:
            clean_database()
            last_clean = time.time()

        try:
            new_blocks = block_filter.get_new_entries()
         
            for block_hash in new_blocks:
                try: 
                    block = w3.eth.get_block(block_hash, full_transactions=True)
                    print(f"📦 Processing Block: {block['number']} ({len(block.transactions)} txs)")

                    for tx in block.transactions:
                        from_addr = tx['from']

                        if from_addr not in processed_addr:
                            print(f"Analyzing new address: {from_addr}")

                            final_vector = get_feature_vector(from_addr, MASTER_SENT_PATH, MASTER_REC_PATH, MASTER_COLUMN_LIST)

                            if final_vector.empty:
                                continue

                            fraud_probability = model.predict_proba(final_vector)[0][1]

                            log_transaction(from_addr, tx['hash'].hex(), fraud_probability)

                            if fraud_probability >= 0.3:
                                print(f"🚨 FRAUD ALERT! (Probability: {fraud_probability})")

                            processed_addr.add(from_addr)

                except Exception as e:
                    print(f"Error processing block: {e}")

            time.sleep(5)

        except Exception as e:
            print(f"Loop error: {e}")
            time.sleep(5)

        time.sleep(1)

def main():
    setup_databse()
    main_loop()

if __name__ == "__main__":
    main()
