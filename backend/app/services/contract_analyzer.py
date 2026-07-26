import os
import anthropic
import json
import sqlite3
import hashlib
from datetime import datetime

from dotenv import load_dotenv
from pathlib import Path

try:
    from .contract_fetcher import fetch_source_code
except ImportError:
    from contract_fetcher import fetch_source_code

base_dir = Path(__file__).parent
env_path = base_dir.parent.parent / ".env"
cache_db_path = base_dir.parent.parent / "database/contract_cache.db"

load_dotenv(env_path)

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# caches previously audited contracts
def init_cache_db():
    conn = sqlite3.connect(cache_db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contract_analysis (
            contract_address TEXT PRIMARY KEY,
            contract_name TEXT,
            safe_score INTEGER,
            risk_level TEXT,
            vulnerabilities TEXT,
            summary TEXT,
            source_code_hash TEXT,
            analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

def get_cached_analysis(contract_address: str):
    init_cache_db()

    conn = sqlite3.connect(cache_db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM contract_analysis
        WHERE contract_address = ?
    ''', (contract_address.lower(),))

    result = cursor.fetchone()
    conn.close()

    if result:
        return {
            "contract_name": result["contract_name"],
            "safe_score": result["safe_score"],
            "risk_level": result["risk_level"],
            "vulnerabilities": json.loads(result["vulnerabilities"]),
            "summary": result["summary"],
            "cached": True,
            "analyzed_at": result["analyzed_at"]
        }

    return None

def save_analysis_to_cache(contract_address: str, analysis_result: dict, source_code: str):
    source_code_hash = hashlib.sha256(source_code.encode()).hexdigest()

    conn = sqlite3.connect(cache_db_path)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT OR REPLACE INTO contract_analysis
        (contract_address, contract_name, safe_score, risk_level, vulnerabilities, summary, source_code_hash, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        contract_address.lower(),
        analysis_result.get("contract_name", "Unknown"),
        analysis_result.get("safe_score", 0),
        analysis_result.get("risk_level", "Unknown"),
        json.dumps(analysis_result.get("vulnerabilities", [])),
        analysis_result.get("summary", ""),
        source_code_hash,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()

def analyze_smart_contract(source_code, contract_address=None, use_cache=True):

    # check cache first if address provided and caching enabled
    if use_cache and contract_address:
        cached_result = get_cached_analysis(contract_address)
        if cached_result:
            print(f"Using cached analysis for {contract_address}")
            return cached_result

    system_prompt = """
    You are an expert Smart Contract Auditor and Blockchain Security Engineer.

    Your task is to audit the provided Solidity source code for security vulnerabilities.
    Look for:
    - Reentrancy attacks
    - Integer Overflow/Underflow (if SafeMath is missing)
    - Unchecked return values
    - Centralization risks (e.g., owner can mint infinite tokens)
    - Honeypots (restrictions on selling)

    You must return your response in STRICT JSON format. Do not include markdown formatting (like ```json).
    Just return the raw JSON string with the following structure:
    {
        "contract_name": "Name of the main contract",
        "safe_score": (integer 0-100, where 100 is perfectly safe, 0 is dangerous),
        "risk_level": "Low" | "Medium" | "High" | "Critical",
        "vulnerabilities": [
            {
                "type": "Type of vulnerability (e.g., Reentrancy)",
                "severity": "High" | "Medium" | "Low",
                "description": "Brief explanation of the issue",
                "line_number": "Approximate line number (or 'N/A')"
            }
        ],
        "summary": "A 2-3 sentence executive summary of the contract's safety."
    }
    """

    try:
        message = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=2000,
            temperature=0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Here is the source code for analysis:\n\n{source_code}"
                },
                {
                    "role": "assistant",
                    "content": "{"
                }
            ]
        )

        raw_response = message.content[0].text
        full_json_string = "{" + raw_response

        analysis_result = json.loads(full_json_string)
        analysis_result["cached"] = False

        # save to cache if address provided
        if contract_address:
            save_analysis_to_cache(contract_address, analysis_result, source_code)

        return analysis_result

    except Exception as e:
        # Log details server-side only — don't leak internals to clients
        print(f"Error analyzing contract: {e}")
        return {"error": "Failed to analyze contract"}
    
def main():
    honeypot_address = "0x34C6211621f2763c60Eb007dC2aE91090A2d22f6"
    uniswap_address = "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984"

    source_code_string = fetch_source_code(honeypot_address)

    print(analyze_smart_contract(source_code_string, contract_address=honeypot_address))


if __name__ == "__main__":
    main()