import os
from dotenv import load_dotenv

import requests
import pandas as pd

from io import StringIO

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

ETHERSCAN_API_KEY = os.environ.get('ETHERSCAN_API_KEY')
url = "https://api.etherscan.io/v2/api"

def fetch_eth_history(address):
    query = {
        "apikey": ETHERSCAN_API_KEY,
        "chainid":"1",
        "module": "account",
        "action": "txlist",
        "address": address,
        "tag":"latest",
        "startblock":"0",
        "endblock":"9999999999",
        "offset":"1000",
        "sort":"desc"
    }

    eth_response = requests.get(url, params = query)
    eth_history_dict = eth_response.json()

    if eth_history_dict['status'] == '1':
        eth_df = pd.json_normalize(eth_history_dict, 'result')
    else:
        return

    eth_csv_path = os.path.join(os.path.dirname(__file__), '..', 'data/' 'eth_history.csv')
    eth_df.to_csv(eth_csv_path, index = False)

    return eth_df

def fetch_erc_20_history(address):
    query = {
        "apikey":ETHERSCAN_API_KEY,
        "chainid":"1",
        "module":"account",
        "action":"tokentx",
        "txhash":"0x15f8e5ea1079d9a0bb04a4c58ae5fe7654b5b2b4463375ff7ffb490aa0032f3a",
        "address": address,
        "tag":"latest",
        "startblock":"0","endblock":
        "9999999999",
        "page":"1",
        "offset":"1",
        "sort":"desc",
        "contractaddress":"0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2"
    }

    erc20_response = requests.get(url, params = query)
    erc20_history_dict = erc20_response.json()

    erc20_csv_path = os.path.join(os.path.dirname(__file__), '..', 'data/' 'erc20_history.csv' )

    erc20_df = pd.json_normalize(erc20_history_dict, 'result')

    erc20_df.to_csv(erc20_csv_path, index = False)

    return erc20_df

def eth_feature_generator(df, address):
    # Cleaning
    df = df[df['isError'] == '0']
    df['timeStamp'] = pd.to_numeric(df['timeStamp'])
    df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='s')
    df['value'] = pd.to_numeric(df['value']) / 10**18

    sent_df = df[df['from'].str.lower() == address.lower()]
    received_df = df[df['to'].str.lower() == address.lower()]

    # Volume & Total Features
    sent_tnx = len(sent_df)
    received_tnx = len(received_df)
    total_transactions = len(df)

    # Time Features
    first_tx_time = df['timeStamp'].min()
    last_tx_time = df['timeStamp'].max()

    if first_tx_time is pd.NaT:
        time_diff_between_first_and_last = 0
    else:
        time_diff_between_first_and_last = (last_tx_time - first_tx_time).total_seconds() / 60

    sorted_sent_df = sent_df.sort_values(by = 'timeStamp')
    sent_diff = sorted_sent_df['timeStamp'].diff()

    if sent_diff.isnull().all():
        avg_avg_min_between_sent_tnx = 0
    else:   
        avg_avg_min_between_sent_tnx = sent_diff.mean().total_seconds() / 60

    sorted_received_df = received_df.sort_values(by = 'timeStamp')
    received_diff = sorted_received_df['timeStamp'].diff()

    if received_diff.isnull().all():
        avg_min_between_received_tnx = 0
    else:
        avg_min_between_received_tnx = received_diff.mean() / 60

    unique_sent_to_addresses = sent_df['to'].nunique()
    unique_received_from_addresses = received_df['from'].nunique()

    # Value features
    total_ether_sent = sent_df['value'].sum()
    total_ether_received = received_df['value'].sum()
    total_ether_balance = total_ether_received - total_ether_sent

    min_val_sent = sent_df['value'].min() if not sent_df.empty else 0
    max_val_sent = sent_df['value'].max() if not sent_df.empty else 0
    avg_val_sent = sent_df['value'].mean() if not sent_df.empty else 0

    min_val_received = received_df['value'].min() if not received_df.empty else 0
    max_val_received = received_df['value'].max() if not received_df.empty else 0
    avg_val_received = received_df['value'].mean() if not received_df.empty else 0

    # Contract Features
    num_created_contracts = df[df['contractAddress'] != '']['contractAddress'].count()
    sent_to_contract_df = sent_df[sent_df['input'].str.lower() != '0x']
    total_ether_sent_to_contract = sent_to_contract_df['value'].sum() if not sent_to_contract_df.empty else 0
    min_val_sent_to_contract = sent_to_contract_df['value'].min() if not sent_to_contract_df.empty else 0
    max_val_sent_to_contract = sent_to_contract_df['value'].max() if not sent_to_contract_df.empty else 0
    avg_val_sent_to_contract = sent_to_contract_df['value'].mean() if not sent_to_contract_df.empty else 0

    eth_feature_dict = {
        'avg_min_between_sent_tnx': avg_avg_min_between_sent_tnx,
        'avg_min_between_received_tnx': avg_min_between_received_tnx,
        'time_diff_between_first_and_last': time_diff_between_first_and_last,
        'sent_tnx': sent_tnx,
        'received_tnx': received_tnx,
        'num_created_contracts': num_created_contracts,
        'unique_received_from_addresses': unique_received_from_addresses,
        'unique_sent_to_addresses': unique_sent_to_addresses,
        'min_value_received': min_val_received,
        'max_value_received': max_val_received,
        'avg_val_received': avg_val_received,
        'min_val_sent': min_val_sent,
        'max_val_sent': max_val_sent,
        'avg_val_sent': avg_val_sent,
        'min_val_sent_to_contract': min_val_sent_to_contract,
        'max_val_sent_to_contract': max_val_sent_to_contract,
        'avg_val_sent_to_contract': avg_val_sent_to_contract, # You'll need to calculate this
        'total_transactions': total_transactions,
        'total_ether_sent': total_ether_sent,
        'total_ether_received': total_ether_received,
        'total_ether_sent_to_contract': total_ether_sent_to_contract,
        'total_ether_balance': total_ether_balance
    }

    eth_df = pd.DataFrame(eth_feature_dict, index=[0])

    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data/' 'eth_features.csv' )
    eth_df.to_csv(csv_path, index = False)

    return eth_df

def erc20_feature_generator():
    return

def main():
    address = "0xbE982C014bC3b3D847782e9Fc1162aB34F260134" # Random placeholder wallet for now

    df = fetch_eth_history(address)
    eth_feature_generator(df, address)


if __name__== "__main__":
    main()

