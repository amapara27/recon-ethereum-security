import os
import requests

from pathlib import Path
from dotenv import load_dotenv

base_dir = Path(__file__).parent
lists_dir = base_dir / "lists"
models_dir = base_dir.parent / "models"
env_path = base_dir.parent.parent / ".env"
alerts_path = base_dir.parent / "alerts.db"

load_dotenv(env_path)

ETHERSCAN_API_KEY = os.environ.get('ETHERSCAN_API_KEY')


