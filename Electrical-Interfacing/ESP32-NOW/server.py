from flask import Flask, request, jsonify
import os
from datetime import datetime

app = Flask(__name__)
LOG_FILE = "sensor_data_log.txt"

@app.route('/api/sensor', methods=['POST'])
def receive_sensor_data():
    try:
        data = request.get_json()

        # Format the data as a log entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = (
            f"[{timestamp}]\n"
            f"RFID: {data.get('rfid')}\n"
            f"Weight: {data.get('weight')} kg\n"
            f"Temperature: {data.get('temperature')} °C\n"
            f"Humidity: {data.get('humidity')} %\n"
            f"Light: {data.get('light')} lx\n"
            f"Pressure: {data.get('pressure')} hPa\n"
            f"{'-'*40}\n"
        )

        # Append to log file
        with open(LOG_FILE, "a") as f:
            f.write(log_entry)

        print("✅ Data received and logged.")
        return jsonify({"message": "Data logged successfully"}), 200

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": "Failed to process data"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
