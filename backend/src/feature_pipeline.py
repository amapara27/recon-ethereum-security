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
        "chainid": "1",
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock":"0",
        "endblock":"9999999999",
        "sort":"desc"
    }

    eth_response = requests.get(url, params = query)
    eth_history_dict = eth_response.json()

    if eth_history_dict['status'] == '1':
        eth_df = pd.json_normalize(eth_history_dict, 'result')
    else:
        return pd.DataFrame()

    eth_csv_path = os.path.join(os.path.dirname(__file__), '..', 'data/' 'eth_history.csv')
    eth_df.to_csv(eth_csv_path, index = False)

    return eth_df

def fetch_erc_20_history(address):
    query = {
        "apikey": ETHERSCAN_API_KEY,
        "module": "account",
        "chainid": "1",
        "action": "tokentx",
        "address": address,
        "startblock": "0",
        "endblock": "9999999999",
        "sort":"asc"
    }

    erc20_response = requests.get(url, params = query)
    erc20_history_dict = erc20_response.json()

    if erc20_history_dict['status'] == '1':
        erc20_df = pd.json_normalize(erc20_history_dict, 'result')
    else:
        print(f"Etherscan API Error for ERC20: {erc20_history_dict['message']}")
        return pd.DataFrame()

    erc20_csv_path = os.path.join(os.path.dirname(__file__), '..', 'data/' 'erc20_history.csv' )
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

    eth_feature_df = pd.DataFrame(eth_feature_dict, index=[0])

    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data/' 'eth_features.csv' )
    eth_feature_df.to_csv(csv_path, index = False)

    return eth_feature_df

def erc20_feature_generator(df, address, sent_vocab_path, rec_vocab_path):
    # Cleaning
    if df.empty:
        print(f"No successful ERC20 transactions found for {address}. Returning empty DataFrame.")
        return pd.DataFrame()

    df['timeStamp'] = pd.to_numeric(df['timeStamp'], errors='coerce')
    df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='s')

    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['tokenDecimal'] = pd.to_numeric(df['tokenDecimal'], errors='coerce')
    
    df = df.dropna(subset=['value', 'tokenDecimal', 'timeStamp'])
    
    df['tokenDecimal'] = df['tokenDecimal'].replace(0, 18) 
    df['real_value'] = df['value'] / (10**df['tokenDecimal'])

    sent_df = df[df['from'].str.lower() == address.lower()]
    received_df = df[df['to'].str.lower() == address.lower()]

    # Volume & Total Features
    ERC20_sent_tnx = len(sent_df)
    ERC20_rec_tnx = len(received_df)
    ERC20_total_tnxs = len(df)

    # Time Features
    ERC20_avg_time_between_sent_tnx = 0
    sorted_sent_df = sent_df.sort_values(by = 'timeStamp')
    sent_diff = sorted_sent_df['timeStamp'].diff()

    if not sent_diff.isnull().all():
        ERC20_avg_time_between_sent_tnx = sent_diff.mean().total_seconds() / 60

    ERC20_avg_time_between_rec_tnx = 0
    sorted_received_df = received_df.sort_values(by = 'timeStamp')
    received_diff = sorted_received_df['timeStamp'].diff()

    if not received_diff.isnull().all():
        # Fix: Variable name mismatch (was ERC20_avg_min_between_rec_tnx)
        ERC20_avg_time_between_rec_tnx = received_diff.mean().total_seconds() / 60

    ERC20_uniq_sent_addr = sent_df['to'].nunique()
    ERC20_uniq_rec_addr = received_df['from'].nunique()

    # Value features
    ERC20_total_ether_sent = sent_df['real_value'].sum()
    ERC20_total_ether_received = received_df['real_value'].sum()

    ERC20_min_val_sent = sent_df['real_value'].min() if not sent_df.empty else 0
    ERC20_max_val_sent = sent_df['real_value'].max() if not sent_df.empty else 0
    ERC20_avg_val_sent = sent_df['real_value'].mean() if not sent_df.empty else 0

    ERC20_min_val_rec = received_df['real_value'].min() if not received_df.empty else 0
    ERC20_max_val_rec = received_df['real_value'].max() if not received_df.empty else 0
    ERC20_avg_val_rec = received_df['real_value'].mean() if not received_df.empty else 0

    # Contract Features
    sent_to_contract_df = sent_df[sent_df['input'].str.lower() != '0x']
    
    ERC20_total_ether_sent_to_contract = sent_to_contract_df['real_value'].sum()
    ERC20_min_val_sent_contract = sent_to_contract_df['real_value'].min() if not sent_to_contract_df.empty else 0
    ERC20_max_val_sent_contract = sent_to_contract_df['real_value'].max() if not sent_to_contract_df.empty else 0
    ERC20_avg_val_sent_contract = sent_to_contract_df['real_value'].mean() if not sent_to_contract_df.empty else 0
    
    ERC20_uniq_sent_contract_addr = sent_to_contract_df['to'].nunique()
    ERC20_uniq_rec_contract_addr = received_df['contractAddress'].nunique()

    ERC20_uniq_sent_token_name = sent_df['tokenName'].nunique()
    ERC20_uniq_rec_token_name = received_df['tokenName'].nunique()

    ERC20_avg_time_between_contract_tnx = 0
    sorted_contract_df = sent_to_contract_df.sort_values(by='timeStamp')
    contract_diff = sorted_contract_df['timeStamp'].diff()
    
    if not contract_diff.isnull().all():
        ERC20_avg_time_between_contract_tnx = contract_diff.mean().total_seconds() / 60

    quant_feature_dict = {
        'ERC20_total_tnxs': ERC20_total_tnxs,
        'ERC20_total_ether_received': ERC20_total_ether_received,
        'ERC20_total_ether_sent': ERC20_total_ether_sent,
        'ERC20_total_ether_sent_to_contract': ERC20_total_ether_sent_to_contract,
        'ERC20_uniq_sent_addr': ERC20_uniq_sent_addr,
        'ERC20_uniq_rec_addr': ERC20_uniq_rec_addr,
        'ERC20_uniq_sent_contract_addr': ERC20_uniq_sent_contract_addr,
        'ERC20_uniq_rec_contract_addr': ERC20_uniq_rec_contract_addr,
        'ERC20_avg_time_between_sent_tnx': ERC20_avg_time_between_sent_tnx,
        'ERC20_avg_time_between_rec_tnx': ERC20_avg_time_between_rec_tnx,
        # 'ERC20_avg_time_between_rec_2_tnx': 0, # Skipping as logic is undefined
        'ERC20_avg_time_between_contract_tnx': ERC20_avg_time_between_contract_tnx,
        'ERC20_min_val_rec': ERC20_min_val_rec,
        'ERC20_max_val_rec': ERC20_max_val_rec,
        'ERC20_avg_val_rec': ERC20_avg_val_rec,
        'ERC20_min_val_sent': ERC20_min_val_sent,
        'ERC20_max_val sent': ERC20_max_val_sent, # Matching CSV typo
        'ERC20_avg_val_sent': ERC20_avg_val_sent,
        'ERC20_min_val_sent_contract': ERC20_min_val_sent_contract,
        'ERC20_max_val_sent_contract': ERC20_max_val_sent_contract,
        'ERC20_avg_val_sent_contract': ERC20_avg_val_sent_contract,
        'ERC20_uniq_sent_token_name': ERC20_uniq_sent_token_name,
        'ERC20_uniq_rec_token_name': ERC20_uniq_rec_token_name
    }
    
    # --- Categorical Features ---
    MASTER_SENT_TOKEN_COLS = load_token_vocabulary(sent_vocab_path)
    MASTER_REC_TOKEN_COLS = load_token_vocabulary(rec_vocab_path)

    if not MASTER_SENT_TOKEN_COLS or not MASTER_REC_TOKEN_COLS:
        print("ERROR: Token vocabularies not loaded. Skipping categorical features.")
        return pd.DataFrame(quant_feature_dict, index=[0])

    most_sent_token = "None"
    if not sent_df.empty:
        try:
            most_sent_token = sent_df['tokenName'].mode()[0]
        except IndexError:
            most_sent_token = "None"

    most_rec_token = "None"
    if not received_df.empty:
        try:
            most_rec_token = received_df['tokenName'].mode()[0]
        except IndexError:
            most_rec_token = "None"

    live_sent_col = f"ERC20_most_sent_token_{most_sent_token}"
    live_rec_col = f"ERC20_most_rec_token_{most_rec_token}"

    sent_token_features = {col_name: 0 for col_name in MASTER_SENT_TOKEN_COLS}
    rec_token_features = {col_name: 0 for col_name in MASTER_REC_TOKEN_COLS}

    if live_sent_col in sent_token_features:
        sent_token_features[live_sent_col] = 1
    else:
        if "ERC20_most_sent_token_None" in sent_token_features:
             sent_token_features["ERC20_most_sent_token_None"] = 1

    if live_rec_col in rec_token_features:
        rec_token_features[live_rec_col] = 1
    else:
        if "ERC20_most_rec_token_None" in rec_token_features:
            rec_token_features["ERC20_most_rec_token_None"] = 1

    final_feature_dict = {
        **quant_feature_dict,
        **sent_token_features,
        **rec_token_features
    }


    erc20_feature_df = pd.DataFrame(final_feature_dict, index=[0])

    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data/' 'erc20_features.csv' )
    erc20_feature_df.to_csv(csv_path, index = False)

    return erc20_feature_df
    
def load_token_vocabulary(filepath: str) -> set:
    """
    Loads the master list of token columns from a text file.
    Returns a set for fast lookups.
    """
    try:
        with open(filepath, 'r') as f:
            # .strip("', ") removes the quotes, commas, and whitespace
            return {line.strip().strip("', ") for line in f if line.strip()}
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Vocabulary file not found at {filepath}")
        return set()

def main():
    address = "0xbE982C014bC3b3D847782e9Fc1162aB34F260134" # Random placeholder wallet for now

    eth_df = fetch_eth_history(address)
    eth_feature_generator(eth_df, address)

    sent_path = "master_sent.txt"
    rec_path = "master_rec.txt"

    erc20_df = fetch_erc_20_history(address)
    erc20_feature_generator(erc20_df, address, sent_path, rec_path)


if __name__== "__main__":
    main()

