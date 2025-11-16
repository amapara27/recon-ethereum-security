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

load_dotenv(env_path)
rpc_url = os.environ.get('ALCHEMY_RPC_URL')

MODEL_PATH = models_dir / "fraud_model.joblib"
MASTER_COLUMN_PATH = lists_dir / "master_column_list.txt"
MASTER_SENT_PATH = lists_dir / "master_sent.txt"
MASTER_REC_PATH = lists_dir / "master_rec.txt"
MASTER_COLUMN_LIST = load_master_column_list(MASTER_COLUMN_PATH)

model = joblib.load(MODEL_PATH)
w3 = Web3(Web3.HTTPProvider(rpc_url))

# # Model Testing
# test_address = "0x41cd1343c6e497c075b7ed1e9f003d65141833e2"
# final_vector_df = get_feature_vector(test_address, MASTER_SENT_PATH, MASTER_REC_PATH, MASTER_COLUMN_LIST)

# # Checking probability, if greater than 0.3 then warned a fraud
# fraud_probability = model.predict_proba(final_vector_df)[0][1]
# prediction = 1 if fraud_probability >= 0.3 else 0
# print(f"Fraud probability: {fraud_probability:.1%}")
# print(f"Prediction: {prediction}" )


processed_addr = set()

def main_loop():
    if not w3.is_connected():
        print("Connection failed. Check your RPC_URL.")
        return
    
    block_filter = w3.eth.filter('latest')

    while True:
        new_blocks = block_filter.get_new_entries()

        for block_hash in new_blocks:
            block = w3.eth.get_block(block_hash, full_transactions=True)

            for tx in block.transactions:
                from_addr = tx['from']

                if from_addr not in processed_addr:
                    print(f"Analyzing new address: {from_addr}")

                    final_vector = get_feature_vector(from_addr, MASTER_SENT_PATH, MASTER_REC_PATH, MASTER_COLUMN_LIST)

                    if final_vector.empty:
                        continue

                    fraud_probability = model.predict_proba(final_vector)[0][1]

                    if fraud_probability >= 0.3:
                        print(f"Fraud Detected (Probability: {fraud_probability})")
                    else:
                        print(f"No Fraud Detected! (Probability: {1 - fraud_probability}")

                    processed_addr.add(from_addr)

        time.sleep(1)

main_loop()

