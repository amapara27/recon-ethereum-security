# Recon: An Ethereum Security Dashboard

Recon is a full-stack security dashboard that provides real-time threat intelligence for the Ethereum network. It combines on-chain data, AI-powered analysis, and off-chain intelligence to create a comprehensive platform for users, developers, and security professionals to identify and react to threats.

The first completed module is a **Real-Time Fraud Detection Engine** that listens to new Ethereum blocks, analyzes the behavioral "fingerprint" of transaction senders, and flags high-risk activity with a predictive model.

---

## 🌎 Real-World Applications

A fully-featured Recon dashboard has direct applications for various users in the Web3 ecosystem:

* **For DeFi Users & Traders:**
    * **Live Risk Feed:** Before interacting with a new protocol, a user can check the "Live Feed" to see if other users' transactions are being flagged as fraudulent, indicating a potential hack or rug pull in progress.
    * **Token Vetting:** Use the "Honeypot Detector" to check an ERC20 token contract *before* buying, protecting against "buy-only" scams.

* **For Security Analysts & Researchers:**
    * **Fund Tracing (AML):** Use the "Wallet Investigator" to trace the flow of stolen funds from a known hack, helping to identify the attacker's wallet clusters and their attempts to launder money through exchanges.
    * **Threat Monitoring:** Actively monitor the network for anomalies and identify the rise of new scam techniques (like address poisoning) as they happen.

* **For Developers & Auditors:**
    * **Pre-Deployment Check:** Use the "AI Smart Contract Auditor" to get an instant "second opinion" on new code, catching common vulnerabilities like re-entrancy or integer overflows *before* the contract is deployed.
    * **Phishing Prevention:** Monitor the "Scam Domain" feed to quickly identify and report new phishing sites that are impersonating their project, protecting their user base.

---

## 🚀 Core Features (Current State)

The foundational **Real-Time Fraud Detection Engine** is complete and functional:

* **Live Transaction Monitoring:** Connects to an Ethereum node (via Infura) and listens for new blocks as they are mined.
* **On-the-Fly Feature Engineering:** For any new address, this system fetches its entire transaction history (both ETH and ERC20) from Etherscan. It then instantly calculates a 770+ feature vector representing the address's complete financial behavior.
* **AI-Powered Scoring:** Uses a pre-trained Random Forest model (trained on a labeled dataset with an F1-score of 0.92) to assign a fraud probability to every new transaction.
* **Custom Thresholding:** Uses `predict_proba` to set a custom 30% probability threshold, prioritizing high recall to catch as many potential fraud cases as possible.
* **Persistent Alerting:** All detected fraudulent transactions are logged to a local SQLite database for persistent storage.
* **Efficient Caching:** A simple in-memory `set` is used to cache processed addresses, preventing redundant Etherscan API calls for the same address within a single session.

---

## 🛠️ How It Works (Architecture)

Recon is a full-stack application with a clear separation of concerns:

### `backend/`
This directory contains the entire Python-based backend.

* **`src/monitor.py`:** The main "engine." This is a 24/7 script that:
    1.  Loads the ML model and all feature templates.
    2.  Connects to `web3.py` and listens for new blocks.
    3.  For each new transaction, calls the `feature_pipeline`.
    4.  Runs the model to get a fraud probability.
    5.  Saves alerts to the database.

* **`src/feature_pipeline.py`:** The "brain" of the operation. This script contains all the logic to:
    1.  Fetch ETH history (`txlist`) and ERC20 history (`tokentx`) from the Etherscan API.
    2.  Perform complex feature generation (e.g., `avg_min_between_sent_tnx`, `real_value` of tokens).
    3.  Generate the 770+ feature one-hot encoded categorical vector.
    4.  Return the final vector, perfectly ordered to match the model's training data.

* **`src/api.py`:** (In Development) A Flask API server that provides a "bridge" between the database and the frontend. It will serve alerts as a JSON endpoint.

* **`data/`:** Contains the original CSV dataset used for training.

* **`models/`:** Contains the final trained `fraud_detector.joblib` model and the "master list" `.txt` files used as templates for the feature pipeline.

### `frontend/`
(In Development) A JavaScript-based dashboard that will visualize the data from the API.

---

## 🔧 Tech Stack

* **Backend:** Python
* **Machine Learning:** Scikit-learn (Random Forest), Pandas, NumPy
* **Blockchain Data:** Web3.py (for live blocks), Etherscan API (for historical data)
* **Database:** SQLite
* **Environment:** Conda
* **API (Planned):** Flask, Flask-CORS

---

## 🏁 How to Run (Current State)

Currently, the backend monitor can be run by itself to start populating the database.

### 1. Setup

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/your-username/recon.git](https://github.com/your-username/recon.git)
    cd recon
    ```

2.  **Create & activate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate eth_fraud_detector
    ```

3.  **Create your environment file:**
    * Create a file named `.env` in the root of the project (`recon/.env`).
    * Add your API keys:
        ```
        INFURA_RPC_URL="https"//mainnet.infura.io/v3/YOUR_KEY_HERE"
        ETHERSCAN_API_KEY="YOUR_KEY_HERE"
        ```

### 2. Run the Monitor

1.  Open your terminal and run the monitor script:
    ```bash
    python backend/src/monitor.py
    ```
2.  The script will initialize the database, connect to Ethereum, and begin listening for new blocks.
3.  You will see alerts printed directly to your console and saved in the `backend/alerts.db` file.

---

## 🗺️ Future Plans (Dashboard Roadmap)

The existing backend is the foundation for the full Recon Security Dashboard. The planned roadmap includes:

* **✅ Full-Stack API/UI:** Complete the `api.py` and build a JavaScript (React/Svelte/Vue) frontend to visualize the live fraud alerts.
* **📈 Wallet Investigator:** A search-and-report tool to analyze any address and visualize its transaction graph (AML/fund tracing).
* **🛡️ AI Smart Contract Auditor:** A new tab where users can paste Solidity code to be scanned for common vulnerabilities (re-entrancy, integer overflows) by an AI model.
* **🍯 Honeypot/Scam Token Detector:** A tool to analyze ERC20 token contracts for signs of malicious code (e.g., un-sellable tokens) or market manipulation (e.g., wash trading).
* **🌐 Phishing/Scam Domain Monitor:** Integrate off-chain data (like new domain registrations) to flag "typosquatted" phishing sites (e.g., "uniswap-claim.xyz").

---

## 📄 License

Distributed under the MIT License. See `LICENSE` file for more information.