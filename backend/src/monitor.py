from web3 import Web3
import os
from dotenv import load_dotenv
from web3 import Web3

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

rpc_url = os.environ.get('ALCHEMY_RPC_URL')

w3 = Web3(Web3.HTTPProvider(rpc_url))

# Check  cnnection
if w3.is_connected():
    print("Success!")
    
    # Get the latest block number
    latest_block = w3.eth.block_number
    print(f"The latest block number is: {latest_block}")

else:
    print("Connection failed. Check your RPC_URL.")