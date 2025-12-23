import os
import anthropic
import json

from dotenv import load_dotenv
from pathlib import Path

from contract_fetcher import fetch_source_code

base_dir = Path(__file__).parent
env_path = base_dir.parent / ".env"

load_dotenv(env_path)

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

def analyze_smart_contract(source_code):
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
        "risk_score": (integer 0-100, where 100 is perfectly safe, 0 is dangerous),
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
        
        return json.loads(full_json_string)

    except Exception as e:
        print(f"Error analyzing contract: {e}")
        return {
            "error": "Failed to analyze contract", 
            "details": str(e)
        }
    
def main():
    honeypot_address = "0x34C6211621f2763c60Eb007dC2aE91090A2d22f6"
    uniswap_address = "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984"

    source_code_string = fetch_source_code(uniswap_address)

    print(analyze_smart_contract(source_code_string))


if __name__ == "__main__":
    main()