# Recon: An Ethereum Security Dashboard

Recon is a full-stack security dashboard that provides real-time threat intelligence for the Ethereum network. It combines on-chain data, AI-powered analysis, and off-chain intelligence to create a comprehensive platform for users, developers, and security professionals to identify and react to threats.

The first completed module is a **Real-Time Fraud Detection Engine** that listens to new Ethereum blocks, analyzes the behavioral "fingerprint" of transaction senders, and flags high-risk activity with a predictive model.

---

## Demo
[![Watch the Demo](https://img.youtube.com/vi/5ofxkW9tX0w/0.jpg)](https://youtu.be/5ofxkW9tX0w)

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

## 🚀 Core Features

* **Real-Time Fraud Detection:** Monitors live Ethereum blocks, analyzes transaction patterns using 770+ features, and flags suspicious activity with AI (Random Forest, F1: 0.92)
* **Smart Feature Engineering:** Automatically fetches complete transaction history from Etherscan and calculates behavioral fingerprints on-the-fly
* **Full-Stack Dashboard:** React frontend with FastAPI backend displaying live fraud alerts and transaction monitoring
* **Production-Ready:** Dockerized deployment with persistent SQLite database and intelligent caching

---

## 🛠️ Architecture

**Backend (`backend/`):**
* `monitor.py` - Listens to live Ethereum blocks via Web3.py, processes transactions through the ML pipeline, and stores alerts
* `feature_pipeline.py` - Fetches transaction history from Etherscan and generates 770+ feature vectors for fraud prediction
* `api.py` - FastAPI server exposing RESTful endpoints for frontend data access
* `models/` - Pre-trained Random Forest classifier and feature templates
* `data/` - Original training dataset

**Frontend (`frontend/`):**
* React dashboard with live fraud alerts, transaction scanner, and responsive UI

---

## 🔧 Tech Stack

* **Backend:** Python, FastAPI
* **Machine Learning:** Scikit-learn (Random Forest), Pandas, NumPy
* **Blockchain:** Web3.py, Etherscan API
* **Frontend:** JavaScript, React
* **Database:** SQLite
* **Deployment:** Docker, Docker Compose

---

## 🏁 Quick Start

### Docker (Recommended)
```bash
# Clone and setup
git clone https://github.com/your-username/recon.git
cd recon

# Add API keys to backend/.env
echo 'INFURA_RPC_URL="https://mainnet.infura.io/v3/YOUR_KEY"' > backend/.env
echo 'ETHERSCAN_API_KEY="YOUR_KEY"' >> backend/.env

# Run
docker-compose up --build
```
Access at `http://localhost:5173` (frontend) and `http://localhost:8000` (API)

### Local Development
```bash
# Backend
conda env create -f environment.yml && conda activate eth_fraud_detector
cd backend && python src/monitor.py & python src/api.py

# Frontend (separate terminal)
cd frontend && npm install && npm run dev
```

---

## 🗺️ Roadmap

**✅ Completed**
* Full-stack fraud detection system with React UI and FastAPI backend
* Dockerized deployment with live transaction tracking and database persistence

**🚧 In Progress**
* AWS Cloud Deployment (EC2, RDS, CloudWatch, Load Balancing)

**📋 Planned**
* Wallet Investigator - Transaction graph visualization and fund tracing
* AI Smart Contract Auditor - Vulnerability detection for Solidity code
* Honeypot/Scam Token Detector - Malicious ERC20 contract analysis
* Phishing Domain Monitor - Real-time scam website detection
* Analytics Dashboard - Historical trends and fraud pattern analysis
* Alert System - Configurable notifications for high-risk activity

---

## 📄 License

Distributed under the MIT License. See `LICENSE` file for more information.
