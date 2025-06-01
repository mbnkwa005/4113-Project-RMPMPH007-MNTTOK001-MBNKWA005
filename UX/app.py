from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import sqlite3
import os
from werkzeug.utils import secure_filename
import torch
from ultralytics import YOLO
import yaml
from PIL import Image
import json
from datetime import datetime
import sys
import pandas as pd
import io
from transformers import OwlViTProcessor, OwlViTForObjectDetection, pipeline
import cv2
import base64
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

print("Starting application...")
app = Flask(__name__)

# In-memory buffer for live detection results
live_detections_buffer = []
# Define maximum buffer size to prevent excessive memory usage
MAX_BUFFER_SIZE = 20 # Keep last 20 live detections in memory

print("Configuring upload folder...")
# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

print("Attempting to load models...")
# Initialize models globally
owl_model = None
yolo_model = None

# Add VGG model class
class VGGModel(nn.Module):
    def __init__(self, num_classes=2):
        super(VGGModel, self).__init__()
        # Load pretrained VGG16
        self.vgg = models.vgg16(pretrained=True)
        
        # Freeze the feature layers
        for param in self.vgg.features.parameters():
            param.requires_grad = False
            
        # Modify the classifier
        self.vgg.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        return self.vgg(x)

def load_models():
    global owl_model, yolo_model
    if owl_model is None:
        try:
            # Load OWL-ViT model and processor
            owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
            owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            owl_model = (owl_model, owl_processor)  # Store both model and processor
            print("OWL-ViT model loaded successfully")
        except Exception as e:
            print(f"Error loading OWL-ViT model: {str(e)}")
            owl_model = None
    
    if yolo_model is None:
        try:
            # Load YOLO model
            yolo_model = YOLO('best.pt')
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {str(e)}")
            import traceback
            traceback.print_exc()
            yolo_model = None
    
    return owl_model, yolo_model

def process_frame(frame, owl_model, yolo_model):
    # Convert frame to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Resize image for faster processing
    max_size = 640
    ratio = min(max_size / pil_image.width, max_size / pil_image.height)
    new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Step 1: Check if it's a penguin using OWL-ViT
    model, processor = owl_model  # Unpack the tuple
    inputs = processor(
        text=["a penguin"],
        images=pil_image,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([pil_image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=0.3
    )[0]
    
    # Check if any penguin is detected
    is_penguin = False
    penguin_confidence = 0.0
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score > 0.3:
            is_penguin = True
            penguin_confidence = float(score)
            break
    
    if is_penguin:
        # Step 2: Check for molting using YOLO
        is_molting, molting_confidence = detect_molting(pil_image)
        
        return {
            'is_penguin': True,
            'is_molting': is_molting,
            'penguin_confidence': penguin_confidence,
            'molting_confidence': molting_confidence,
            'detections': [{
                'label': 'molting' if is_molting else 'not_molting',
                'confidence': molting_confidence
            }]
        }
    else:
        return {
            'is_penguin': False,
            'is_molting': False,
            'penguin_confidence': penguin_confidence,
            'molting_confidence': 0.0,
            'detections': []
        }

@app.route('/')
def index():
    try:
        conn = get_db_connection()
        # Fetch all penguins for the index page, ordered by last update time
        penguins = conn.execute('SELECT * FROM penguins ORDER BY last_update_time DESC').fetchall()
        conn.close()
        
        # Convert penguins to list of dicts and parse predictions
        penguin_list = []
        if penguins:
            for penguin in penguins:
                penguin_dict = dict(penguin)
                # Parse predictions
                if penguin_dict['predictions']:
                    try:
                        predictions = json.loads(penguin_dict['predictions'])
                        # Sort predictions by timestamp if available
                        if predictions and len(predictions) > 0:
                            predictions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                        penguin_dict['predictions'] = predictions
                        # Check for non-penguin detection
                        if predictions and len(predictions) > 0:
                            latest_prediction = predictions[0]  # Use first item since we sorted in reverse
                            if latest_prediction.get('label') == 'non_penguin':
                                penguin_dict['is_danger'] = True
                                penguin_dict['danger_message'] = latest_prediction.get('warning', 'DANGER: Non-penguin object detected!')
                            else:
                                penguin_dict['is_danger'] = False
                                penguin_dict['danger_message'] = None
                    except Exception as e:
                        print(f"Error parsing predictions for penguin {penguin_dict.get('name', 'Unknown')}: {str(e)}")
                        penguin_dict['predictions'] = []
                        penguin_dict['is_danger'] = False
                        penguin_dict['danger_message'] = None
                else:
                    penguin_dict['predictions'] = []
                    penguin_dict['is_danger'] = False
                    penguin_dict['danger_message'] = None
                
                # Parse weight history
                if penguin_dict['weight_history']:
                    try:
                        penguin_dict['weight_history'] = json.loads(penguin_dict['weight_history'])
                    except:
                        penguin_dict['weight_history'] = []
                
                # Parse molting history
                if penguin_dict['molting_history']:
                    try:
                        penguin_dict['molting_history'] = json.loads(penguin_dict['molting_history'])
                    except:
                        penguin_dict['molting_history'] = []
                
                penguin_list.append(penguin_dict)
        
        return render_template('index.html', penguins=penguin_list)
    except Exception as e:
        print(f"Error in index route: {str(e)}")
        import traceback
        traceback.print_exc()
        return str(e), 500

@app.route('/search')
def search():
    query = request.args.get('q', '')
    conn = get_db_connection()
    penguins = conn.execute('''
        SELECT * FROM penguins 
        WHERE name LIKE ? 
        OR tags LIKE ?
    ''', (f'%{query}%', f'%{query}%')).fetchall()
    conn.close()
    return jsonify([dict(penguin) for penguin in penguins])

@app.route('/stats')
def stats():
    conn = get_db_connection()

    # Get total penguins
    total = conn.execute('SELECT COUNT(*) as count FROM penguins').fetchone()['count']

    # Get average weight
    avg_data = conn.execute('SELECT AVG(weight) as avg_weight FROM penguins').fetchone()
    avg_weight = round(avg_data['avg_weight'], 2) if avg_data['avg_weight'] else 0

    conn.close()

    return jsonify({
        'total_penguins': total,
        'avg_weight': avg_weight,
    })

def detect_penguin(image_path):
    """
    First step: Detect if the object is a penguin using OWL-ViT
    Returns: (is_penguin, confidence)
    """
    if not owl_model or not yolo_model:
        print("Error: Models not loaded")
        return False, 0.0

    try:
        # Load and preprocess image
        image = Image.open(image_path)
        model, processor = owl_model  # Unpack the tuple
        
        # Prepare inputs
        inputs = processor(
            text=["a penguin"],
            images=image,
            return_tensors="pt"
        )
        
        # Run inference
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model(**inputs)
        
        # Get predictions
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.1
        )[0]

        # Check if any detection has high confidence
        is_penguin = False
        confidence = 0.0
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > 0.5:  # Confidence threshold
                is_penguin = True
                confidence = float(score)
                break

        print(f"\nPenguin Detection Results:")
        print(f"Is Penguin: {'Yes' if is_penguin else 'No'}")
        print(f"Confidence: {confidence * 100:.2f}%")

        return is_penguin, confidence

    except Exception as e:
        print(f"Error in penguin detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0.0

def detect_molting(image_path):
    """
    Second step: If it's a penguin, detect molting status using YOLO
    Returns: (is_molting, confidence)
    """
    if not yolo_model:
        print("Error: YOLO model not loaded")
        return False, 0.0

    try:
        # Run YOLO prediction
        results = yolo_model(image_path)
        
        # Get the first result (since we're processing one image)
        if len(results) > 0:
            result = results[0]
            
            # Print all detections for debugging
            print("\nYOLO Detection Results:")
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = result.names[class_id]
                print(f"Detected: {label} with confidence: {confidence:.2f}")
            
            # Get the detection with highest confidence
            if len(result.boxes) > 0:
                best_box = result.boxes[0]  # Boxes are already sorted by confidence
                class_id = int(best_box.cls[0])
                confidence = float(best_box.conf[0])
                label = result.names[class_id]
                
                # Check if the detected class is 'molting'
                is_molting = label == 'molting'
                
                print(f"\nBest Detection:")
                print(f"Label: {label}")
                print(f"Confidence: {confidence:.2f}")
                print(f"Is Molting: {is_molting}")
                
                return is_molting, confidence
            else:
                print("No detections found")
                return False, 0.0
        else:
            print("No results from YOLO model")
            return False, 0.0

    except Exception as e:
        print(f"Error in molting detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0.0

@app.route('/add_penguin', methods=['POST'])
def add_penguin():
    if request.method == 'POST':
        try:
            name = request.form['name']
            rfid = request.form['rfid']
            weight = float(request.form['weight'])
            detection_time = request.form['detection_time']
            sex = request.form['sex']
            
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
                
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
                
            if file:
                filename = secure_filename(file.filename)
                file_path = f"uploads/{filename}"
                full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(full_path)
                
                # First check if it's a penguin using OWL-ViT
                is_penguin, penguin_confidence = detect_penguin(full_path)
                
                if not is_penguin:
                    warning = "DANGER: Non-penguin object detected!"
                    prediction = {
                        'label': 'non_penguin',
                        'confidence': penguin_confidence,
                        'timestamp': detection_time,
                        'warning': warning
                    }
                else:
                    # If it's a penguin, check for molting using YOLO
                    is_molting, confidence = detect_molting(full_path)
                    prediction = {
                        'label': 'molting' if is_molting else 'not_molting',
                        'confidence': confidence,
                        'timestamp': detection_time
                    }
                    warning = None
                
                # Initialize weight history with current weight
                weight_history = json.dumps([{
                    'weight': weight,
                    'timestamp': detection_time
                }])
                
                # Initialize predictions with the first detection
                predictions = json.dumps([prediction])
                
                # Initialize molting history if prediction indicates molting
                molting_history = json.dumps([{
                    'probability': confidence if is_penguin else penguin_confidence,
                    'timestamp': detection_time,
                    'is_penguin': is_penguin
                }])
                
                conn = get_db_connection()
                conn.execute('''
                    INSERT INTO penguins (
                        name, rfid, weight, tags, 
                        predictions, weight_history, molting_history,
                        last_update_time, image_path, sex, warning, detection_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    name, rfid, weight, '',
                    predictions, weight_history, molting_history,
                    detection_time, file_path, sex, warning, 'Manual'
                ))
                conn.commit()
                conn.close()
                
                return redirect(url_for('index'))
                
        except Exception as e:
            return jsonify({'error': str(e)}), 400
            
    return jsonify({'error': 'Invalid request method'}), 405

@app.route('/get_penguins', methods=['GET'])
def get_penguins():
    conn = get_db_connection()
    penguins = conn.execute('SELECT id, name, weight, tags, image_path, predictions FROM penguins').fetchall()
    conn.close()
    
    penguin_list = []
    for penguin in penguins:
        predictions = penguin['predictions']
        if predictions:
            try:
                predictions = json.loads(predictions)
            except:
                predictions = None
                
        penguin_list.append({
            'id': penguin['id'],
            'name': penguin['name'],
            'weight': penguin['weight'],
            'tags': penguin['tags'],
            'image_path': penguin['image_path'],
            'predictions': predictions
        })
    
    return jsonify(penguin_list)

@app.route('/get_penguin/<int:penguin_id>', methods=['GET'])
def get_penguin(penguin_id):
    conn = get_db_connection()
    penguin = conn.execute('SELECT id, name, weight, tags, image_path, predictions FROM penguins WHERE id = ?', (penguin_id,)).fetchone()
    conn.close()
    
    if penguin is None:
        return jsonify({'error': 'Penguin not found'}), 404
    
    return jsonify({
        'id': penguin['id'],
        'name': penguin['name'],
        'weight': penguin['weight'],
        'tags': penguin['tags'],
        'image_path': penguin['image_path'],
        'predictions': penguin['predictions']
    })

@app.route('/predict_penguin/<int:penguin_id>', methods=['GET'])
def predict_penguin(penguin_id):
    try:
        conn = get_db_connection()
        penguin = conn.execute('SELECT image_path, predictions FROM penguins WHERE id = ?', (penguin_id,)).fetchone()
        
        if penguin is None:
            return jsonify({'error': 'Penguin not found'}), 404
            
        # Get the full path to the image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(penguin['image_path']))
        
        # Run YOLO prediction if model is available
        predictions = []
        if yolo_model is not None:
            results = yolo_model(image_path)
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    label = result.names[class_id]
                    predictions.append({
                        'label': label,
                        'confidence': confidence,
                        'timestamp': datetime.now().isoformat()
                    })
        else:
            # Return dummy prediction if model is not available
            predictions.append({
                'label': 'Model not available',
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            })
        
        # Get existing predictions
        existing_predictions = []
        if penguin['predictions']:
            try:
                existing_predictions = json.loads(penguin['predictions'])
            except:
                existing_predictions = []
        
        # Add new predictions
        if predictions:
            best_prediction = max(predictions, key=lambda x: x['confidence'])
            existing_predictions.append(best_prediction)
            
            # Store all predictions in the database
            conn.execute('UPDATE penguins SET predictions = ? WHERE id = ?',
                       (json.dumps(existing_predictions), penguin_id))
            conn.commit()
        
        conn.close()
        return jsonify({'predictions': existing_predictions})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_weight_change(weight_history, new_weight):
    if not weight_history:
        return 0, False, 'none'
    
    try:
        weight_history = json.loads(weight_history)
        if not weight_history:
            return 0, False, 'none'
            
        last_weight = weight_history[-1]['weight']
        weight_change = new_weight - last_weight
        weight_change_percent = (weight_change / last_weight) * 100
        
        # Determine severity based on percentage change
        abs_change = abs(weight_change_percent)
        if abs_change > 25:
            severity = 'severe'  # Red warning
            is_significant = True
        elif abs_change > 10:
            severity = 'moderate'  # Yellow warning
            is_significant = True
        else:
            severity = 'none'  # No warning
            is_significant = False
        
        return weight_change_percent, is_significant, severity
    except:
        return 0, False, 'none'

@app.route('/update_weight/<int:penguin_id>', methods=['POST'])
def update_weight(penguin_id):
    try:
        data = request.get_json()
        new_weight = float(data['weight'])
        
        conn = get_db_connection()
        penguin = conn.execute('SELECT weight_history FROM penguins WHERE id = ?', (penguin_id,)).fetchone()
        
        if penguin is None:
            return jsonify({'error': 'Penguin not found'}), 404
            
        # Calculate weight change
        weight_change_percent, is_significant, severity = calculate_weight_change(penguin['weight_history'], new_weight)
        
        # Get existing weight history
        weight_history = []
        if penguin['weight_history']:
            try:
                weight_history = json.loads(penguin['weight_history'])
            except:
                weight_history = []
        
        # Add new weight record with change information
        weight_history.append({
            'weight': new_weight,
            'timestamp': datetime.now().isoformat(),
            'weight_change_percent': weight_change_percent,
            'is_significant': is_significant,
            'severity': severity
        })
        
        # Update penguin's current weight and history
        conn.execute('''
            UPDATE penguins 
            SET weight = ?, weight_history = ? 
            WHERE id = ?
        ''', (new_weight, json.dumps(weight_history), penguin_id))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True, 
            'weight_history': weight_history,
            'weight_change_percent': weight_change_percent,
            'is_significant': is_significant,
            'severity': severity
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/penguin_stats/<int:penguin_id>')
def penguin_stats(penguin_id):
    conn = get_db_connection()
    penguin = conn.execute('SELECT * FROM penguins WHERE id = ?', (penguin_id,)).fetchone()
    conn.close()
    
    if penguin is None:
        return jsonify({'error': 'Penguin not found'}), 404
    
    # Parse predictions and weight history
    predictions = []
    weight_history = []
    
    if penguin['predictions']:
        try:
            predictions = json.loads(penguin['predictions'])
        except:
            predictions = []
            
    if penguin['weight_history']:
        try:
            weight_history = json.loads(penguin['weight_history'])
        except:
            weight_history = []
    
    return jsonify({
        'id': penguin['id'],
        'name': penguin['name'],
        'current_weight': penguin['weight'],
        'predictions': predictions,
        'weight_history': weight_history
    })

@app.route('/penguin/<int:penguin_id>')
def penguin_profile(penguin_id):
    try:
        conn = get_db_connection()
        penguin = conn.execute('SELECT * FROM penguins WHERE id = ?', (penguin_id,)).fetchone()
        conn.close()
        
        if penguin is None:
            return redirect(url_for('index'))
        
        # Parse JSON fields
        penguin_dict = dict(penguin)
        
        # Parse predictions
        if penguin_dict['predictions']:
            try:
                predictions = json.loads(penguin_dict['predictions'])
                penguin_dict['predictions'] = predictions
                # Check for non-penguin detection
                if predictions and len(predictions) > 0:
                    latest_prediction = predictions[-1]
                    if latest_prediction.get('label') == 'non_penguin':
                        penguin_dict['is_danger'] = True
                        penguin_dict['danger_message'] = latest_prediction.get('warning', 'DANGER: Non-penguin object detected!')
                    else:
                        penguin_dict['is_danger'] = False
                        penguin_dict['danger_message'] = None
            except:
                penguin_dict['predictions'] = []
                penguin_dict['is_danger'] = False
                penguin_dict['danger_message'] = None
        
        # Parse weight history
        if penguin_dict['weight_history']:
            try:
                penguin_dict['weight_history'] = json.loads(penguin_dict['weight_history'])
            except:
                penguin_dict['weight_history'] = []
        
        # Parse molting history
        if penguin_dict['molting_history']:
            try:
                penguin_dict['molting_history'] = json.loads(penguin_dict['molting_history'])
            except:
                penguin_dict['molting_history'] = []
        
        return render_template('penguin_profile.html', penguin=penguin_dict)
    except Exception as e:
        print(f"Error in penguin_profile route: {str(e)}")
        return str(e), 500

@app.route('/update_penguin/<int:penguin_id>', methods=['POST'])
def update_penguin(penguin_id):
    if request.method == 'POST':
        try:
            weight = float(request.form.get('weight', 0))
            
            # Handle image upload if provided
            image_path = None
            prediction = None
            if 'image' in request.files:
                file = request.files['image']
                if file.filename != '':
                    filename = secure_filename(file.filename)
                    file_path = f"uploads/{filename}"
                    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(full_path)
                    image_path = file_path
                    
                    # Run YOLO prediction on new image
                    if yolo_model:
                        try:
                            results = yolo_model(full_path)
                            if results and len(results) > 0:
                                result = results[0]
                                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                                    box = result.boxes[0]
                                    class_id = int(box.cls[0])
                                    confidence = float(box.conf[0])
                                    label = result.names[class_id]
                                    prediction = {
                                        'label': label,
                                        'confidence': confidence,
                                        'timestamp': datetime.now().isoformat()
                                    }
                        except Exception as e:
                            print(f"Error in YOLO prediction: {str(e)}")
                            prediction = {
                                'label': 'Error in detection',
                                'confidence': 0.0,
                                'timestamp': datetime.now().isoformat()
                            }
            
            conn = get_db_connection()
            
            # Update weight history
            weight_history = conn.execute('SELECT weight_history FROM penguins WHERE id = ?', (penguin_id,)).fetchone()['weight_history']
            if weight_history:
                weight_history = json.loads(weight_history)
            else:
                weight_history = []
            
            # Calculate weight change and significance
            weight_change_percent, is_significant, severity = calculate_weight_change(json.dumps(weight_history), weight)
            
            weight_history.append({
                'weight': weight,
                'timestamp': datetime.now().isoformat(),
                'weight_change_percent': weight_change_percent,
                'is_significant': is_significant,
                'severity': severity
            })
            
            # Update predictions if new image was uploaded
            predictions = conn.execute('SELECT predictions FROM penguins WHERE id = ?', (penguin_id,)).fetchone()['predictions']
            if predictions:
                predictions = json.loads(predictions)
            else:
                predictions = []
                
            if prediction:
                predictions.append(prediction)
            
            # Update molting history if prediction was made
            molting_history = conn.execute('SELECT molting_history FROM penguins WHERE id = ?', (penguin_id,)).fetchone()['molting_history']
            if molting_history:
                molting_history = json.loads(molting_history)
            else:
                molting_history = []
                
            if prediction and prediction['label'] == 'molting':
                molting_history.append({
                    'probability': prediction['confidence'],
                    'timestamp': prediction['timestamp']
                })
            
            # Update database
            update_query = '''
                UPDATE penguins 
                SET weight = ?,
                    weight_history = ?,
                    predictions = ?,
                    molting_history = ?,
                    last_update_time = CURRENT_TIMESTAMP
            '''
            params = [weight, json.dumps(weight_history), json.dumps(predictions), json.dumps(molting_history)]
            
            if image_path:
                update_query += ', image_path = ?'
                params.append(image_path)
            
            update_query += ' WHERE id = ?'
            params.append(penguin_id)
            
            conn.execute(update_query, params)
            conn.commit()
            conn.close()
            
            return redirect(url_for('penguin_profile', penguin_id=penguin_id))
            
        except Exception as e:
            return jsonify({'error': str(e)}), 400
            
    return jsonify({'error': 'Invalid request method'}), 405

@app.route('/live_detection')
def live_detection():
    print("Live detection route accessed")
    try:
        return render_template('live_detection.html')
    except Exception as e:
        print(f"Error rendering live_detection template: {str(e)}")
        return str(e), 500

@app.route('/api/esp_data', methods=['POST'])
def receive_esp_data():
    if request.method == 'POST':
        try:
            global live_detections_buffer
            data = request.get_json()
            print("\n" + "="*50)
            print("Received data from ESP:")
            print(data)
            print("="*50 + "\n")

            # Validate required fields
            # Check if either 'rfid' or 'uid' is present
            if 'rfid' not in data and 'uid' not in data:
                return jsonify({'error': 'Missing required field: rfid or uid'}), 400
            
            # Ensure 'weight' is present
            if 'weight' not in data:
                 return jsonify({'error': 'Missing required field: weight'}), 400

            # Process the data and add to live buffer
            processed_data = {
                'rfid': data.get('uid', data.get('rfid')), # Use 'uid' if present, otherwise use 'rfid'
                'name': data.get('name', 'N/A'),
                'weight': float(data['weight']),
                'tags': data.get('tags', ''),
                'image_path': data.get('image_path', 'default.jpg'),
                'label': data.get('label', 'Unknown'),
                'confidence': data.get('confidence', 0.0),
                'timestamp': datetime.now().isoformat(),
                'detection_type': 'ESP'
            }

            # Add to live buffer
            live_detections_buffer.append(processed_data)
            # Keep buffer size limited
            while len(live_detections_buffer) > MAX_BUFFER_SIZE:
                live_detections_buffer.pop(0)  # Remove oldest entry

            print(f"[receive_esp_data] Added ESP data to buffer. Buffer size: {len(live_detections_buffer)}")

            return jsonify({
                'success': True,
                'message': 'ESP data received successfully'
            })

        except Exception as e:
            print(f"Error in receive_esp_data: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': 'An internal error occurred.'}), 500

@app.route('/check_for_new_data', methods=['POST'])
def check_for_new_data():
    if request.method == 'POST':
        try:
            global live_detections_buffer
            data = request.get_json()
            last_checked_time_str = data.get('last_checked_time')

            # If this is a test connection attempt
            if data.get('test'):
                # For live view, we can return current buffer and clear it on test
                current_buffer = list(live_detections_buffer)  # Get a copy
                live_detections_buffer = []  # Clear the buffer after sending
                print(f"[check_for_new_data] Returning {len(current_buffer)} buffered live detections.")
                return jsonify({'new_data': current_buffer})

            # Check the buffer for new data
            if live_detections_buffer:
                current_buffer = list(live_detections_buffer)  # Get a copy
                live_detections_buffer = []  # Clear the buffer after sending
                print(f"[check_for_new_data] Returning {len(current_buffer)} buffered live detections.")
                return jsonify({'new_data': current_buffer})

            # If no new data in buffer, return empty list
            return jsonify({'new_data': []})

        except Exception as e:
            print(f"Error in check_for_new_data: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': 'An internal error occurred.'}), 500

    return jsonify({'error': 'Invalid request method'}), 405

@app.route('/database')
def database_view():
    try:
        conn = get_db_connection()
        penguins = conn.execute('SELECT * FROM penguins').fetchall()
        conn.close()
        
        # Convert penguins to list of dicts and parse JSON fields
        penguin_list = []
        for penguin in penguins:
            penguin_dict = dict(penguin)
            
            # Parse predictions
            if penguin_dict['predictions']:
                try:
                    predictions = json.loads(penguin_dict['predictions'])
                    penguin_dict['predictions'] = predictions
                    # Check for non-penguin detection
                    if predictions and len(predictions) > 0:
                        latest_prediction = predictions[-1]
                        if latest_prediction.get('label') == 'non_penguin':
                            penguin_dict['is_danger'] = True
                            penguin_dict['danger_message'] = latest_prediction.get('warning', 'DANGER: Non-penguin object detected!')
                        else:
                            penguin_dict['is_danger'] = False
                            penguin_dict['danger_message'] = None
                except:
                    penguin_dict['predictions'] = []
                    penguin_dict['is_danger'] = False
                    penguin_dict['danger_message'] = None
            
            # Parse weight history
            if penguin_dict['weight_history']:
                try:
                    penguin_dict['weight_history'] = json.loads(penguin_dict['weight_history'])
                except:
                    penguin_dict['weight_history'] = []
            
            # Parse molting history
            if penguin_dict['molting_history']:
                try:
                    penguin_dict['molting_history'] = json.loads(penguin_dict['molting_history'])
                except:
                    penguin_dict['molting_history'] = []
            
            penguin_list.append(penguin_dict)
        
        return render_template('database.html', penguins=penguin_list)
    except Exception as e:
        print(f"Error in database_view route: {str(e)}")
        return str(e), 500

@app.route('/download_excel')
def download_excel():
    conn = get_db_connection()
    # Get all penguins data
    penguins = conn.execute('SELECT * FROM penguins').fetchall()
    conn.close()
    
    # Convert to pandas DataFrame
    df = pd.DataFrame([dict(penguin) for penguin in penguins])
    
    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Penguins', index=False)
    
    output.seek(0)
    
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='penguin_database.xlsx'
    )

@app.route('/api/update_penguin_data', methods=['POST'])
def api_update_penguin_data():
    # Log connection attempt
    client_ip = request.remote_addr
    print("\n" + "="*50)
    print(f"Connection attempt from IP: {client_ip}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Request Method: {request.method}")
    print(f"Request Headers: {dict(request.headers)}")
    print("="*50)

    if request.method == 'POST':
        try:
            global live_detections_buffer
            data = request.get_json()
            print("\nReceived data:")
            print(json.dumps(data, indent=2))
            print("="*50 + "\n")

            # Validate required fields - check for either rfid or uid
            if 'rfid' not in data and 'uid' not in data:
                print("Error: Missing required field: rfid or uid")
                return jsonify({'error': 'Missing required field: rfid or uid'}), 400
            
            # Ensure 'weight' is present
            if 'weight' not in data:
                print("Error: Missing required field: weight")
                return jsonify({'error': 'Missing required field: weight'}), 400

            # Get the identifier (either rfid or uid)
            identifier = data.get('uid', data.get('rfid'))

            # Get or create penguin based on identifier
            conn = get_db_connection()
            penguin = conn.execute('SELECT * FROM penguins WHERE rfid = ?', (identifier,)).fetchone()

            image_path = None
            prediction = None
            molting_entry = None
            processed_detection_data = None
            warning = None

            # Handle image and run two-step detection if provided
            if 'image' in data:
                try:
                    print("Processing image data...")
                    import base64
                    from io import BytesIO
                    image_data = base64.b64decode(data['image'])
                    filename = f"{identifier}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    file_path = f"uploads/{filename}"
                    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                    # Save image
                    with open(full_path, 'wb') as f:
                        f.write(image_data)
                    image_path = file_path
                    print(f"Image saved as: {file_path}")

                    # Step 1: Check if it's a penguin using OWL-ViT
                    print("Running penguin detection...")
                    is_penguin, penguin_confidence = detect_penguin(full_path)
                    print(f"Penguin detection result: {is_penguin} (confidence: {penguin_confidence:.2f})")
                    
                    if not is_penguin:
                        warning = "DANGER: Non-penguin object detected!"
                        prediction = {
                            'label': 'non_penguin',
                            'confidence': penguin_confidence,
                            'timestamp': datetime.now().isoformat(),
                            'warning': warning
                        }
                        print("Non-penguin detected!")
                    else:
                        # Step 2: If it's a penguin, check for molting using YOLO
                        print("Running molting detection...")
                        is_molting, molting_confidence = detect_molting(full_path)
                        prediction = {
                            'label': 'molting' if is_molting else 'not_molting',
                            'confidence': molting_confidence,
                            'timestamp': datetime.now().isoformat()
                        }
                        print(f"Molting detection result: {is_molting} (confidence: {molting_confidence:.2f})")
                        
                        if is_molting:
                            molting_entry = {
                                'probability': molting_confidence,
                                'timestamp': datetime.now().isoformat()
                            }

                    # Prepare data to add to live buffer and potentially save to DB
                    processed_detection_data = {
                        'rfid': identifier,
                        'name': data.get('name', penguin['name'] if penguin else 'N/A'),
                        'weight': float(data['weight']),
                        'tags': data.get('tags', penguin['tags'] if penguin else ''),
                        'image_path': image_path,
                        'label': prediction.get('label', 'Unknown'),
                        'confidence': prediction.get('confidence', 0.0),
                        'timestamp': prediction.get('timestamp', datetime.now().isoformat()),
                        'detection_type': 'Live',
                        'warning': warning
                    }
                    print("Detection data processed successfully")

                except Exception as e:
                    print(f"Error processing image or running detection: {str(e)}")
                    import traceback
                    traceback.print_exc()

            if penguin is None:
                print(f"Creating new penguin with identifier: {identifier}")
                # Create new penguin if identifier not found
                if 'name' not in data:
                    print("Error: Name required for new penguin")
                    return jsonify({'error': 'Name required for new penguin'}), 400

                # Initialize weight history
                weight_history = json.dumps([{
                    'weight': float(data['weight']),
                    'timestamp': datetime.now().isoformat()
                }])

                # Insert new penguin
                conn.execute('''
                    INSERT INTO penguins (
                        name, rfid, weight, tags,
                        predictions, weight_history, molting_history,
                        last_update_time, image_path, sex, warning, detection_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['name'],
                    identifier,
                    float(data['weight']),
                    data.get('tags', ''),
                    json.dumps([prediction]) if prediction else '[]',
                    weight_history,
                    json.dumps([molting_entry]) if molting_entry else '[]',
                    datetime.now().isoformat(),
                    image_path or 'default.jpg',
                    data.get('sex', 'Unknown'),
                    warning,
                    'Live'
                ))
                conn.commit()
                print("New penguin created successfully")

                # Add the processed data to the live buffer if successful
                if processed_detection_data:
                    live_detections_buffer.append(processed_detection_data)
                    # Keep buffer size limited
                    while len(live_detections_buffer) > MAX_BUFFER_SIZE:
                        live_detections_buffer.pop(0)
                    print(f"Added new penguin data to buffer. Buffer size: {len(live_detections_buffer)}")

            else:
                print(f"Updating existing penguin with identifier: {identifier}")
                # Update existing penguin
                weight = float(data['weight'])

                # Calculate weight change
                weight_change_percent, is_significant, severity = calculate_weight_change(
                    penguin['weight_history'],
                    weight
                )

                # Update weight history
                weight_history = []
                if penguin['weight_history']:
                    try:
                        weight_history = json.loads(penguin['weight_history'])
                    except:
                        weight_history = []

                weight_history.append({
                    'weight': weight,
                    'timestamp': datetime.now().isoformat(),
                    'weight_change_percent': weight_change_percent,
                    'is_significant': is_significant,
                    'severity': severity
                })

                # Update predictions history if a new prediction was made
                predictions_history = []
                if penguin['predictions']:
                    try:
                        predictions_history = json.loads(penguin['predictions'])
                    except:
                        predictions_history = []

                if prediction:
                    predictions_history.append(prediction)

                # Update molting history if prediction was made and indicates molting
                molting_history = []
                if penguin['molting_history']:
                    try:
                        molting_history = json.loads(penguin['molting_history'])
                    except:
                        molting_history = []

                if molting_entry:
                    molting_history.append(molting_entry)

                # Update database
                update_query = '''
                    UPDATE penguins
                    SET weight = ?,
                        weight_history = ?,
                        predictions = ?,
                        molting_history = ?,
                        last_update_time = CURRENT_TIMESTAMP,
                        detection_type = ?,
                        warning = ?
                '''
                params = [weight, json.dumps(weight_history), json.dumps(predictions_history), 
                         json.dumps(molting_history), 'Live', warning]

                # Add image_path to update if a new image was processed
                if image_path:
                    update_query += ', image_path = ?'
                    params.append(image_path)

                update_query += ' WHERE rfid = ?'
                params.append(identifier)

                conn.execute(update_query, params)
                conn.commit()
                print("Penguin data updated successfully")

                # Add the processed data to the live buffer if successful
                if processed_detection_data:
                    live_detections_buffer.append(processed_detection_data)
                    # Keep buffer size limited
                    while len(live_detections_buffer) > MAX_BUFFER_SIZE:
                        live_detections_buffer.pop(0)
                    print(f"Added existing penguin data to buffer. Buffer size: {len(live_detections_buffer)}")

            conn.close()

            print("\nRequest processed successfully!")
            print("="*50 + "\n")

            return jsonify({
                'success': True,
                'message': 'Penguin data updated successfully',
                'rfid': identifier,
                'detection_result': prediction if prediction else None
            })

        except Exception as e:
            print(f"Error in api_update_penguin_data: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': 'An internal error occurred.'}), 500

    return jsonify({'error': 'Invalid request method'}), 405

@app.route('/delete_penguin/<int:penguin_id>', methods=['POST'])
def delete_penguin(penguin_id):
    conn = get_db_connection()
    
    # Get image path before deleting the record
    penguin = conn.execute('SELECT image_path FROM penguins WHERE id = ?', (penguin_id,)).fetchone()
    image_path = penguin['image_path'] if penguin else None

    conn.execute('DELETE FROM penguins WHERE id = ?', (penguin_id,))
    conn.commit()
    conn.close()

    # Delete associated image file if it exists and is not the default
    if image_path and image_path != 'default.jpg':
        full_image_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(image_path))
        if os.path.exists(full_image_path):
            try:
                os.remove(full_image_path)
                print(f"Deleted image file: {full_image_path}")
            except Exception as e:
                print(f"Error deleting image file {full_image_path}: {str(e)}")

    return redirect(url_for('database_view'))

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Get the image data from the request
        image_data = request.get_json().get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
            
        # Remove the data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
            
        # Load models if not already loaded
        owl_model, yolo_model = load_models()
        
        # Process the frame
        results = process_frame(frame, owl_model, yolo_model)
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error in detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_db_connection():
    conn = sqlite3.connect('penguins.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    
    # Create table if it doesn't exist
    conn.execute('''
        CREATE TABLE IF NOT EXISTS penguins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            rfid TEXT UNIQUE,
            weight REAL NOT NULL,
            tags TEXT NOT NULL,
            image_path TEXT NOT NULL,
            predictions TEXT,
            weight_history TEXT,
            molting_history TEXT,
            sex TEXT,
            warning TEXT,
            detection_type TEXT DEFAULT "Unknown",
            last_update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Check if warning column exists, if not add it
    try:
        conn.execute('SELECT warning FROM penguins LIMIT 1')
    except sqlite3.OperationalError:
        print("Adding warning column to penguins table...")
        conn.execute('ALTER TABLE penguins ADD COLUMN warning TEXT')

    # Check if detection_type column exists, if not add it
    try:
        conn.execute('SELECT detection_type FROM penguins LIMIT 1')
    except sqlite3.OperationalError:
        print("Adding detection_type column to penguins table...")
        conn.execute('ALTER TABLE penguins ADD COLUMN detection_type TEXT DEFAULT "Unknown"')

    conn.commit()
    conn.close()

# Add custom filter for JSON parsing
@app.template_filter('from_json')
def from_json(value):
    try:
        return json.loads(value)
    except:
        return []

# Initialize models at startup
load_models()

if __name__ == '__main__':
    init_db()
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print("\n" + "="*50)
    print("Server is running!")
    print("Local endpoint: http://localhost:5000/api/update_penguin_data")
    print("Network endpoint: http://{}:5000/api/update_penguin_data".format(local_ip))
    print("Debug mode: ON")
    print("Host: 0.0.0.0 (accepting connections from all interfaces)")
    print("Port: 5000")
    print("="*50 + "\n")
    
    # Add more detailed error handling
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        print("Please check if port 5000 is already in use or blocked by firewall") 