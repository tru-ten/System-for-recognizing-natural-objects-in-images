from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Завантаження моделі
model = YOLO('best.pt')

@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        # Отримати зображення з запиту
        file = request.files.get('image')
        
        if not file:
            return jsonify({'error': 'No image provided'}), 400
        
        # Прочитати зображення
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Виконати детекцію
        results = model(img)[0]
        
        # Намалювати bounding boxes
        annotated_img = results.plot()
        
        # Конвертувати назад у base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Підготувати результати
        detections = []
        class_counts = {}
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            confidence_score = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            detections.append({
                'class': cls_name,
                'confidence': round(confidence_score, 3),
                'bbox': [round(x1), round(y1), round(x2), round(y2)]
            })
            
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{img_base64}',
            'detections': detections,
            'class_counts': class_counts,
            'total_objects': len(detections),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Інформація про модель"""
    return jsonify({
        'classes': list(model.names.values()),
        'model_type': 'YOLO',
        'num_classes': len(model.names)
    })

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route("/about")
def about():
    return send_from_directory('.', 'about.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)