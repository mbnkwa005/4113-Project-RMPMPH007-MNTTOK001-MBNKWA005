from flask import Flask, request, jsonify
from datetime import datetime
import os

app = Flask(__name__)

# Configuration
LOG_FILE = 'esp32_data.log'
MAX_LOG_SIZE = 1024 * 1024  # 1MB max log size

def ensure_log_file():
    """Create log file if it doesn't exist"""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w') as f:
            f.write("ESP32 Data Log - Created at {}\n\n".format(datetime.now().isoformat()))

def rotate_log_if_needed():
    """Rotate log file if it gets too large"""
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > MAX_LOG_SIZE:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"esp32_data_{timestamp}.log"
        os.rename(LOG_FILE, backup_name)
        ensure_log_file()

def log_data(data):
    """Log data to file with timestamp"""
    timestamp = datetime.now().isoformat()
    with open(LOG_FILE, 'a') as f:
        f.write(f"[{timestamp}] {data}\n")

@app.route('/api/data', methods=['POST'])
def receive_data():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data received"}), 400
        
        print(f"Received data from ESP32: {data}")
        
        # Log the data to file
        rotate_log_if_needed()
        log_data(str(data))
        
        return jsonify({"status": "success", "message": "Data received and logged"}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/data', methods=['GET'])
def get_data():
    """Return the last 100 lines from the log file"""
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()[-100:]  # Get last 100 lines
        return jsonify({"data": lines}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/logs', methods=['GET'])
def get_full_log():
    """Return the complete log file"""
    try:
        with open(LOG_FILE, 'r') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/plain'}
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    ensure_log_file()
    app.run(host='0.0.0.0', port=5000, debug=True)