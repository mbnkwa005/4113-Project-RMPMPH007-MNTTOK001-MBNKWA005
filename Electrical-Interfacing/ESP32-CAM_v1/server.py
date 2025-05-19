from flask import Flask, request, jsonify
import os
from datetime import datetime

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if not request.data:
            return jsonify({"error": "No data received"}), 400
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"esp32_{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save raw image data
        with open(filepath, 'wb') as f:
            f.write(request.data)
        
        return jsonify({
            "status": "success",
            "message": "Image saved",
            "filename": filename,
            "path": filepath
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)