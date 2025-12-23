import os
import requests

from pathlib import Path
from dotenv import load_dotenv

base_dir = Path(__file__).parent
lists_dir = base_dir / "lists"
models_dir = base_dir.parent / "models"
env_path = base_dir.parent / ".env"
alerts_path = base_dir.parent / "alerts.db"

load_dotenv(env_path)

ETHERSCAN_API_KEY = os.environ.get('ETHERSCAN_API_KEY')

url = "https://api.etherscan.io/v2/api"

def fetch_source_code(address):
    verify_query = {
        "apikey": ETHERSCAN_API_KEY,
        "chainid": "1",
        "module": "contract",
        "action": "getabi",
        "address": address,
    }

    verify_response = requests.get(url, params = verify_query)
    verify_response = verify_response.json()

    if verify_response.get("status") == 0:
        return "High Risk / Unknown"
    
    sc_query = {
        "apikey": ETHERSCAN_API_KEY,
        "chainid": "1",
        "module": "contract",
        "action": "getsourcecode",
        "address": address,
    }

    sc_response = requests.get(url, params = sc_query)
    sc_response = sc_response.json()

    result = sc_response.get("result")

    return result[0]["SourceCode"]

def main():
    print(fetch_source_code("0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984"))

if __name__ == "__main__":
    main()