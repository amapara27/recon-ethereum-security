# Recon: Ethereum Blockchain Dashboard

Recon is a **Real-Time Fraud Detection Engine** that listens to live Ethereum blocks, analyzes the behavioral "fingerprint" of transaction senders, and flags high-risk activity using machine learning. Built with Python, FastAPI, and React, it provides a production-ready system for monitoring Ethereum transactions in real-time.

---

## Demo
[![Watch the Demo](https://img.youtube.com/vi/5ofxkW9tX0w/0.jpg)](https://youtu.be/5ofxkW9tX0w)

## 🌎 Use Cases

* **DeFi Security:** Monitor live transactions for fraudulent patterns before interacting with protocols
* **Threat Detection:** Identify suspicious wallet behavior and potential scams in real-time
* **Research & Analysis:** Study fraud patterns and behavioral fingerprints on the Ethereum network
* **Transaction Monitoring:** Track and analyze transaction flows with ML-powered risk scoring

---

## 🚀 Core Features

* **Real-Time Fraud Detection:** Monitors live Ethereum blocks, analyzes transaction patterns using 770+ features, and flags suspicious activity with AI (Random Forest, F1: 0.92)
* **Smart Contract Auditor:** AI-powered vulnerability scanner using Claude Opus 4.5 to detect reentrancy, honeypots, centralization risks, and other security issues in Solidity contracts
* **Intelligent Caching:** SQL-based caching system for contract analyses to minimize API costs and improve response times
* **Smart Feature Engineering:** Automatically fetches complete transaction history from Etherscan and calculates behavioral fingerprints on-the-fly
* **Full-Stack Dashboard:** React frontend with FastAPI backend displaying live fraud alerts, transaction monitoring, and contract analysis
* **Cloud-Deployed:** Production deployment on AWS EC2 with Docker containerization for 24/7 monitoring
* **Scalable Infrastructure:** Persistent database, intelligent caching, and auto-restart capabilities

---

## 🛠️ Architecture

**Backend (`backend/`):**
* `monitor.py` - Listens to live Ethereum blocks via Web3.py, processes transactions through the ML pipeline, and stores alerts
* `feature_pipeline.py` - Fetches transaction history from Etherscan and generates 770+ feature vectors for fraud prediction
* `contract_analyzer.py` - AI-powered smart contract security auditor with SQL caching
* `contract_fetcher.py` - Retrieves verified contract source code from Etherscan
* `api.py` - FastAPI server exposing RESTful endpoints for fraud alerts and contract analysis
* `models/` - Pre-trained Random Forest classifier and feature templates
* `data/` - Original training dataset
* `alerts.db` - SQLite database storing fraud alerts
* `contract_cache.db` - SQLite database caching contract analysis results

**Frontend (`frontend/`):**
* React dashboard with live fraud alerts, transaction scanner, and responsive UI

---

## 🔧 Tech Stack

* **Backend:** Python, FastAPI
* **Machine Learning:** Scikit-learn (Random Forest), Pandas, NumPy
* **AI:** Anthropic Claude Opus 4.5 for smart contract security analysis
* **Blockchain:** Web3.py, Etherscan API
* **Frontend:** JavaScript, React
* **Database:** SQLite with intelligent caching
* **Deployment:** Docker, Docker Compose, AWS EC2
* **Infrastructure:** Systemd services, persistent volumes, environment-based configuration

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
echo 'ANTHROPIC_API_KEY="YOUR_KEY"' >> backend/.env

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

### AWS EC2 Deployment
```bash
# Launch EC2 instance (Ubuntu 22.04, t2.micro or larger)
# Configure security group: Allow inbound on port 8000

# SSH into instance and setup
sudo apt update && sudo apt install -y docker.io docker-compose git
sudo usermod -aG docker ubuntu

# Clone and configure
git clone https://github.com/your-username/recon.git
cd recon
echo 'INFURA_RPC_URL="https://mainnet.infura.io/v3/YOUR_KEY"' > backend/.env
echo 'ETHERSCAN_API_KEY="YOUR_KEY"' >> backend/.env
echo 'ANTHROPIC_API_KEY="YOUR_KEY"' >> backend/.env

# Deploy with auto-restart
docker-compose up -d
```

Access at `http://YOUR_EC2_IP:8000/docs` for API and `http://YOUR_EC2_IP:8000/alerts` for alerts

---

## 🗺️ Roadmap

**✅ Completed**
* Full-stack fraud detection system with React UI and FastAPI backend
* Dockerized deployment with live transaction tracking and database persistence
* AWS EC2 production deployment with 24/7 monitoring capabilities
* Docker containerization with systemd service management
* AI-powered smart contract security auditor using Claude Opus 4.5
* SQL-based caching system for contract analyses

**📋 Future Enhancements**
* **Wallet Watcher:** Real-time wallet analysis, trade monitoring, portfolio tracking, and behavioral analytics
* **One-Click Staking (Testnet):** Simplified staking interface for testing and development
* **AI Command Bar:** Natural language interface for sending transactions, querying blockchain data, and executing operations
* **Advanced Infrastructure:** RDS migration, CloudWatch monitoring, auto-scaling, load balancing
* **Wallet Investigator:** Transaction graph visualization and fund tracing
* **Token Analysis:** Honeypot and scam token detector for ERC20 contracts
* **Phishing Detection:** Real-time monitoring for scam domains and phishing sites
* **Analytics Dashboard:** Historical trends, fraud pattern analysis, and statistics
* **Alert System:** Configurable notifications and webhooks for high-risk activity

---

## 📄 License

Distributed under the MIT License. See `LICENSE` file for more information.
