# Recon: Ethereum Fraud Detection System

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
* **Smart Feature Engineering:** Automatically fetches complete transaction history from Etherscan and calculates behavioral fingerprints on-the-fly
* **Full-Stack Dashboard:** React frontend with FastAPI backend displaying live fraud alerts and transaction monitoring
* **Cloud-Deployed:** Production deployment on AWS EC2 with Docker containerization for 24/7 monitoring
* **Scalable Infrastructure:** Persistent database, intelligent caching, and auto-restart capabilities

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

**📋 Future Enhancements**
* **Advanced Infrastructure:** RDS migration, CloudWatch monitoring, auto-scaling, load balancing
* **Wallet Investigator:** Transaction graph visualization and fund tracing
* **Smart Contract Auditor:** AI-powered vulnerability detection for Solidity code
* **Token Analysis:** Honeypot and scam token detector for ERC20 contracts
* **Phishing Detection:** Real-time monitoring for scam domains and phishing sites
* **Analytics Dashboard:** Historical trends, fraud pattern analysis, and statistics
* **Alert System:** Configurable notifications and webhooks for high-risk activity

---

## 📄 License

Distributed under the MIT License. See `LICENSE` file for more information.
