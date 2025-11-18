from flask import Flask, jsonify
from flask_cors import CORS
from pathlib import Path
import sqlite3

base_dir = Path(__file__).parent
alerts_path = base_dir.parent / "alerts.db"

app = Flask(__name__)
CORS(app)

def get_db_connection():
    conn = sqlite3.connect(alerts_path)
    conn.row_factory = sqlite3.Row
    return conn

@app.route("/api/get-alerts", methods = ['GET'])
def get_latest_alerts():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        entries = cursor.execute('''
            SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 50
            ''')
        
        alerts = []
        
        for entry in entries:
            alert = {
                'id' : entry['id'],
                'address' : entry['address'],
                'timestamp' : entry['timestamp'],
                'tx_hash' : entry['tx_hash'],
                'probability' : entry['probability']
            }

            alerts.append(alert)

        return jsonify(alerts)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        conn.close()

    

if __name__ == "__main__":
    app.run(debug=True, port=5000)
