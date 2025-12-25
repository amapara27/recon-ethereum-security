<div align="center">

# 🛡️ Recon

### Real-Time Ethereum Fraud Detection & Smart Contract Security Auditor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/react-19.2.0-61dafb.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)

**Recon** is a production-ready, real-time fraud detection engine that monitors live Ethereum blocks, analyzes transaction behavioral fingerprints using machine learning, and audits smart contracts for security vulnerabilities with AI-powered analysis.

[Watch Demo](https://youtu.be/5ofxkW9tX0w) • [Report Bug](https://github.com/your-username/recon/issues) • [Request Feature](https://github.com/your-username/recon/issues)

</div>

---

## 📺 Demo

[![Watch the Demo](https://img.youtube.com/vi/5ofxkW9tX0w/0.jpg)](https://youtu.be/5ofxkW9tX0w)

<div align="center"><i>Click to watch the full demonstration</i></div>

## 🎯 Use Cases

<table>
<tr>
<td width="50%">

**🔐 DeFi Security**
Monitor live transactions for fraudulent patterns before interacting with protocols

</td>
<td width="50%">

**⚠️ Threat Detection**
Identify suspicious wallet behavior and potential scams in real-time

</td>
</tr>
<tr>
<td>

**📊 Research & Analysis**
Study fraud patterns and behavioral fingerprints on the Ethereum network

</td>
<td>

**📈 Transaction Monitoring**
Track and analyze transaction flows with ML-powered risk scoring

</td>
</tr>
</table>

---

## ✨ Key Features

### 🤖 Real-Time Fraud Detection
Monitors live Ethereum blocks and analyzes transaction patterns using **770+ behavioral features** to flag suspicious activity with high accuracy (Random Forest classifier, **F1-Score: 0.92**).

### 🔍 AI-Powered Smart Contract Auditor
Leverages **Claude Opus 4.5** to perform deep security analysis of Solidity contracts, detecting:
- Reentrancy vulnerabilities
- Honeypot schemes
- Centralization risks
- Access control issues
- And more security threats

### ⚡ Intelligent Caching System
SQL-based caching layer that stores contract analyses, reducing API costs and improving response times for repeat queries.

### 🧠 Advanced Feature Engineering
Automatically fetches complete transaction histories from Etherscan and generates comprehensive behavioral fingerprints on-the-fly for accurate fraud prediction.

### 🎨 Modern Full-Stack Dashboard
React-based frontend with FastAPI backend providing:
- Live fraud alerts
- Transaction scanner interface
- Contract security analysis
- Responsive, glassmorphic UI

### ☁️ Production-Ready Deployment
Fully containerized with Docker and Docker Compose, ready for deployment on AWS EC2 or any cloud provider with:
- 24/7 monitoring capabilities
- Persistent SQLite databases
- Auto-restart mechanisms
- Environment-based configuration

---

## 🏗️ Architecture

### Backend (`backend/`)

| Component | Description |
|-----------|-------------|
| **`monitor.py`** | Real-time block listener using Web3.py that processes transactions through the ML pipeline and stores alerts |
| **`feature_pipeline.py`** | Transaction history fetcher that generates 770+ behavioral feature vectors from Etherscan data |
| **`contract_analyzer.py`** | AI-powered security auditor with SQL-based caching for efficient contract analysis |
| **`contract_fetcher.py`** | Verified contract source code retrieval from Etherscan API |
| **`api.py`** | FastAPI REST server exposing endpoints for fraud alerts and contract analysis |
| **`models/`** | Pre-trained Random Forest classifier (F1: 0.92) and feature templates |
| **`data/`** | Original training dataset for model development |
| **`alerts.db`** | SQLite database for persistent fraud alert storage |
| **`contract_cache.db`** | SQLite cache for contract analysis results |

### Frontend (`frontend/`)

Built with **React 19** and **Vite**, featuring:
- Glassmorphic design system
- Real-time fraud alert dashboard
- Transaction scanner interface
- Smart contract auditor UI
- Responsive layout with Bootstrap integration

---

## 🔧 Tech Stack

<table>
<tr>
<td valign="top" width="33%">

### Backend
- **Framework:** FastAPI
- **Language:** Python 3.8+
- **ML/AI:** scikit-learn, Anthropic Claude
- **Blockchain:** Web3.py, Etherscan API
- **Data Processing:** Pandas, NumPy

</td>
<td valign="top" width="33%">

### Frontend
- **Framework:** React 19
- **Build Tool:** Vite
- **Styling:** Bootstrap, Custom CSS
- **HTTP Client:** Axios
- **Type Safety:** ESLint

</td>
<td valign="top" width="33%">

### Infrastructure
- **Containerization:** Docker, Docker Compose
- **Database:** SQLite
- **Deployment:** AWS EC2
- **Process Management:** Auto-restart policies

</td>
</tr>
</table>

---

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have:
- **Docker & Docker Compose** (recommended) or **Python 3.8+** and **Node.js 16+**
- **Infura API Key** ([Get free tier](https://infura.io/))
- **Etherscan API Key** ([Get free tier](https://etherscan.io/apis))
- **Anthropic API Key** ([Get API access](https://console.anthropic.com/))

---

### 🐳 Docker Deployment (Recommended)

The fastest way to get Recon running:

```bash
# 1. Clone the repository
git clone https://github.com/your-username/recon.git
cd recon

# 2. Configure environment variables
cat > backend/.env << EOF
INFURA_RPC_URL="https://mainnet.infura.io/v3/YOUR_INFURA_KEY"
ETHERSCAN_API_KEY="YOUR_ETHERSCAN_KEY"
ANTHROPIC_API_KEY="YOUR_ANTHROPIC_KEY"
EOF

# 3. Launch with Docker Compose
docker-compose up --build
```

**Access Points:**
- 🎨 Frontend Dashboard: `http://localhost:5173`
- 🔌 API Documentation: `http://localhost:8000/docs`
- 📊 Fraud Alerts: `http://localhost:8000/alerts`

---

### 💻 Local Development

For development without Docker:

#### Backend Setup
```bash
# Create conda environment
conda create -n recon python=3.8
conda activate recon

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp backend/.env.example backend/.env
# Edit backend/.env with your API keys

# Start monitoring service (Terminal 1)
cd backend
python src/monitor.py

# Start API server (Terminal 2)
cd backend
python src/api.py
```

#### Frontend Setup
```bash
# Install dependencies (Terminal 3)
cd frontend
npm install

# Start development server
npm run dev
```

**Access:**
- Frontend: `http://localhost:5173`
- API: `http://localhost:8000`

---

### ☁️ AWS EC2 Production Deployment

Deploy Recon to AWS for 24/7 monitoring:

#### 1. Launch EC2 Instance
- **AMI:** Ubuntu Server 22.04 LTS
- **Instance Type:** t2.small or larger (t2.micro may be insufficient)
- **Security Group:**
  - Inbound: Port 8000 (API), Port 22 (SSH)
  - Outbound: All traffic

#### 2. SSH and Install Dependencies
```bash
# Connect to your instance
ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP

# Update system and install Docker
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose git
sudo usermod -aG docker ubuntu
newgrp docker
```

#### 3. Deploy Application
```bash
# Clone repository
git clone https://github.com/your-username/recon.git
cd recon

# Configure environment
cat > backend/.env << EOF
INFURA_RPC_URL="https://mainnet.infura.io/v3/YOUR_INFURA_KEY"
ETHERSCAN_API_KEY="YOUR_ETHERSCAN_KEY"
ANTHROPIC_API_KEY="YOUR_ANTHROPIC_KEY"
EOF

# Start services in detached mode
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f
```

#### 4. Access Your Deployment
- 🔌 API Documentation: `http://YOUR_EC2_PUBLIC_IP:8000/docs`
- 📊 Fraud Alerts Endpoint: `http://YOUR_EC2_PUBLIC_IP:8000/alerts`

#### Optional: Enable HTTPS
For production use, consider setting up Nginx with Let's Encrypt SSL certificates.

---

## 📋 API Endpoints

### Fraud Detection
```http
GET /alerts
```
Returns all detected fraud alerts with transaction details and risk scores.

### Smart Contract Analysis
```http
POST /analyze-contract
Content-Type: application/json

{
  "contract_address": "0x..."
}
```
Performs AI-powered security audit of the specified contract.

### Documentation
```http
GET /docs
```
Interactive Swagger UI documentation for all API endpoints.

---

## 🗺️ Roadmap

### ✅ Completed Features

- [x] Real-time fraud detection with ML (F1: 0.92)
- [x] AI-powered smart contract security auditor
- [x] Full-stack dashboard (React + FastAPI)
- [x] Docker containerization with compose
- [x] AWS EC2 production deployment
- [x] SQL-based intelligent caching
- [x] 770+ behavioral feature engineering
- [x] Persistent database storage

### 🚧 In Progress

- [ ] Enhanced analytics dashboard with historical trends
- [ ] Advanced alert notification system

### 📅 Planned Features

#### Security & Analysis
- **Wallet Investigator**: Transaction graph visualization and fund flow tracing
- **Token Scanner**: Honeypot and scam token detector for ERC20 contracts
- **Phishing Detector**: Real-time monitoring for scam domains and phishing sites
- **Pattern Analyzer**: Advanced fraud pattern analysis and statistics

#### Wallet Tools
- **Wallet Watcher**: Real-time wallet analysis and behavioral tracking
- **Portfolio Monitor**: Trade monitoring and portfolio analytics
- **Transaction Tracker**: Multi-wallet transaction monitoring

#### Infrastructure
- **Database Migration**: Move from SQLite to PostgreSQL/RDS
- **Cloud Monitoring**: CloudWatch integration and alerting
- **Auto-Scaling**: Load balancing and horizontal scaling
- **Alert Webhooks**: Configurable notifications and custom integrations

#### User Experience
- **AI Command Bar**: Natural language interface for blockchain queries
- **One-Click Staking**: Simplified testnet staking interface
- **Custom Dashboards**: User-configurable monitoring views

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Web3.py](https://web3py.readthedocs.io/) - Ethereum blockchain interaction
- [Etherscan API](https://etherscan.io/apis) - Transaction history data
- [Anthropic Claude](https://www.anthropic.com/) - AI-powered contract analysis
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [React](https://reactjs.org/) - Frontend framework

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ for the Ethereum community

[Report Bug](https://github.com/your-username/recon/issues) • [Request Feature](https://github.com/your-username/recon/issues)

</div>
