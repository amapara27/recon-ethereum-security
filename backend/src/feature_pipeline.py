import os
from dotenv import load_dotenv

import requests
import pandas as pd

from io import StringIO

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

ETHERSCAN_API_KEY = os.environ.get('ETHERSCAN_API_KEY')
url = "https://api.etherscan.io/v2/api"

def get_address_history(address):
    query= {
                "apikey": ETHERSCAN_API_KEY,
                "chainid":"1",
                "module": "account",
                "action": "txlist",
                "address":"0xc5102fE9359FD9a28f877a67E36B0F050d81a3CC",
                "tag":"latest",
                "startblock":"0",
                "endblock":"9999999999",
                "page":"1",""
                "offset":"1",
                "sort":"desc"
            }

    response = requests.get(url, params = query)
    history_dict = response.json()

    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data/' 'history.csv')

    df = pd.json_normalize(history_dict, 'result')
    df.to_csv(csv_path, index = False)

    return df

def get_address_features(data):
    return



