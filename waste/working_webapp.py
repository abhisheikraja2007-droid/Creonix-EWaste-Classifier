#!/usr/bin/env python3
"""
Working E-Waste Classification Dashboard
Simplified version that works reliably
"""
from flask import Flask, render_template_string, jsonify, request
import subprocess
import threading
import json
import time
import os

app = Flask(__name__)

# Global variables
realtime_process = None
detection_running = False

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🔋 E-Waste Classification Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 15px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .header { 
            background: linear-gradient(135deg, #4CAF50, #45a049); 
            color: white; 
            padding: 30px; 
            text-align: center; 
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        
        .content { padding: 30px; }
        
        .section { 
            margin-bottom: 30px; 
            padding: 20px; 
            border: 1px solid #e0e0e0; 
            border-radius: 10px; 
            background: #f9f9f9;
        }
        
        .section h2 { 
            color: #333; 
            margin-bottom: 15px; 
            font-size: 1.5em;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        
        .classes-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; 
            margin-top: 15px;
        }
        
        .class-item { 
            background: #4CAF50; 
            color: white; 
            padding: 15px; 
            border-radius: 8px; 
            text-align: center; 
            font-weight: bold;
            transition: transform 0.2s;
        }
        .class-item:hover { transform: translateY(-2px); }
        
        .controls { 
            text-align: center; 
            margin: 20px 0; 
        }
        
        .button { 
            background: #4CAF50; 
            color: white; 
            padding: 15px 30px; 
            border: none; 
            border-radius: 8px; 
            cursor: pointer; 
            font-size: 16px; 
            margin: 10px; 
            transition: all 0.3s;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .button:hover { 
            background: #45a049; 
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }
        .button:disabled { 
            background: #ccc; 
            cursor: not-allowed; 
            transform: none;
        }
        
        .stop-button { background: #f44336; }
        .stop-button:hover { background: #da190b; }
        
        .status { 
            margin: 20px 0; 
            padding: 15px; 
            border-radius: 8px; 
            font-weight: bold;
        }
        .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .status.info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        
        .predictions { 
            max-height: 400px; 
            overflow-y: auto; 
            border: 1px solid #ddd; 
            border-radius: 8px; 
            padding: 15px;
            background: white;
        }
        
        .prediction-item { 
            padding: 10px; 
            margin: 5px 0; 
            border-radius: 5px; 
            border-left: 4px solid;
        }
        .prediction-item.recycle { 
            background: #d4edda; 
            border-left-color: #28a745; 
            color: #155724;
        }
        .prediction-item.waste { 
            background: #f8d7da; 
            border-left-color: #dc3545; 
            color: #721c24;
        }
        
        .camera-placeholder {
            background: #f0f0f0;
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            color: #666;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔋 E-Waste Classification Dashboard</h1>
            <p>Real-time AI-powered waste classification system</p>
        </div>
        
        <div class="content">
            <!-- Trained Classes Section -->
            <div class="section">
                <h2>🎯 Trained Classes</h2>
                <p>Our AI model can classify these e-waste items:</p>
                <div class="classes-grid" id="classesGrid">
                    <!-- Classes will be loaded here -->
                </div>
            </div>
            
            <!-- Camera Section -->
            <div class="section">
                <h2>📹 Camera Feed</h2>
                <div class="camera-placeholder">
                    <h3>📷 Camera Ready</h3>
                    <p>Click "Start Detection" to begin real-time classification</p>
                    <p><small>Camera feed is handled by the detection script</small></p>
                </div>
            </div>
            
            <!-- Controls Section -->
            <div class="section">
                <h2>🎮 Controls</h2>
                <div class="controls">
                    <button class="button" id="startBtn" onclick="startDetection()">🚀 Start Detection</button>
                    <button class="button stop-button" id="stopBtn" onclick="stopDetection()" disabled>⏹️ Stop Detection</button>
                </div>
                <div id="status"></div>
            </div>
            
            <!-- Predictions Section -->
            <div class="section">
                <h2>📊 Real-time Predictions</h2>
                <div class="predictions" id="predictions">
                    <p style="text-align: center; color: #666; padding: 20px;">
                        No predictions yet. Start detection to see real-time results.
                    </p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let detectionRunning = false;
        
        // Load classes on page load
        window.onload = function() {
            loadClasses();
        };
        
        function loadClasses() {
            fetch('/api/classes')
                .then(response => response.json())
                .then(data => {
                    const classesGrid = document.getElementById('classesGrid');
                    classesGrid.innerHTML = '';
                    data.classes.forEach(className => {
                        const classItem = document.createElement('div');
                        classItem.className = 'class-item';
                        classItem.textContent = className;
                        classesGrid.appendChild(classItem);
                    });
                })
                .catch(error => {
                    console.error('Error loading classes:', error);
                });
        }
        
        function startDetection() {
            if (detectionRunning) return;
            
            detectionRunning = true;
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            
            showStatus('🚀 Starting detection...', 'info');
            
            fetch('/start_detection', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showStatus('✅ Detection started successfully!', 'success');
                        startPolling();
                    } else {
                        showStatus('❌ Failed to start detection: ' + data.message, 'error');
                        resetButtons();
                    }
                })
                .catch(error => {
                    showStatus('❌ Error starting detection: ' + error, 'error');
                    resetButtons();
                });
        }
        
        function stopDetection() {
            if (!detectionRunning) return;
            
            detectionRunning = false;
            showStatus('⏹️ Stopping detection...', 'info');
            
            fetch('/stop_detection', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    showStatus('✅ Detection stopped', 'success');
                    resetButtons();
                })
                .catch(error => {
                    showStatus('❌ Error stopping detection: ' + error, 'error');
                    resetButtons();
                });
        }
        
        function resetButtons() {
            detectionRunning = false;
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        }
        
        function showStatus(message, type) {
            const statusDiv = document.getElementById('status');
            statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
        }
        
        function addPrediction(prediction) {
            const predictionsDiv = document.getElementById('predictions');
            
            // Clear "no predictions" message
            if (predictionsDiv.children.length === 1 && predictionsDiv.children[0].tagName === 'P') {
                predictionsDiv.innerHTML = '';
            }
            
            const predictionItem = document.createElement('div');
            predictionItem.className = `prediction-item ${prediction.decision}`;
            predictionItem.innerHTML = `
                <strong>${prediction.class_name}</strong> 
                <span style="float: right;">${prediction.confidence}%</span>
                <br>
                <small>Decision: ${prediction.decision.toUpperCase()}</small>
                <br>
                <small>Time: ${prediction.timestamp}</small>
            `;
            
            predictionsDiv.insertBefore(predictionItem, predictionsDiv.firstChild);
            
            // Keep only last 10 predictions
            while (predictionsDiv.children.length > 10) {
                predictionsDiv.removeChild(predictionsDiv.lastChild);
            }
        }
        
        function startPolling() {
            if (!detectionRunning) return;
            
            fetch('/get_predictions')
                .then(response => response.json())
                .then(data => {
                    if (data.predictions && data.predictions.length > 0) {
                        data.predictions.forEach(prediction => {
                            addPrediction(prediction);
                        });
                    }
                })
                .catch(error => {
                    console.error('Error polling predictions:', error);
                })
                .finally(() => {
                    if (detectionRunning) {
                        setTimeout(startPolling, 1000); // Poll every second
                    }
                });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/classes')
def get_classes():
    return jsonify({
        'classes': ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']
    })

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global realtime_process, detection_running
    
    try:
        if realtime_process and realtime_process.poll() is None:
            return jsonify({'success': False, 'message': 'Detection already running'})
        
        # Start realtime_cls.py
        cmd = [
            'python', 'realtime_cls.py',
            '--weights', 'runs/cls/ewaste/weights/best.pt',
            '--source', '0',
            '--imgsz', '224',
            '--decision-map', 'decision_map.yaml',
            '--min-conf', '0.60',
            '--stable-frames', '3',
            '--trigger', 'none'
        ]
        
        print(f"🚀 Starting detection: {' '.join(cmd)}")
        
        realtime_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Start monitoring thread
        threading.Thread(target=monitor_predictions, daemon=True).start()
        
        detection_running = True
        time.sleep(2)  # Give it time to start
        
        if realtime_process.poll() is None:
            return jsonify({'success': True, 'message': 'Detection started successfully'})
        else:
            stdout, stderr = realtime_process.communicate()
            return jsonify({'success': False, 'message': f'Failed to start: {stderr}'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global realtime_process, detection_running
    
    try:
        if realtime_process and realtime_process.poll() is None:
            realtime_process.terminate()
            realtime_process.wait(timeout=5)
        
        realtime_process = None
        detection_running = False
        
        return jsonify({'success': True, 'message': 'Detection stopped'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# Store predictions
predictions = []

def monitor_predictions():
    global realtime_process, predictions
    
    if not realtime_process:
        return
    
    try:
        for line in iter(realtime_process.stdout.readline, ''):
            if line.strip():
                print(f"📝 Output: {line.strip()}")
                if line.startswith("PREDICTION:"):
                    try:
                        prediction_json = line.replace("PREDICTION:", "").strip()
                        prediction_data = json.loads(prediction_json)
                        predictions.append(prediction_data)
                        print(f"🎯 Prediction: {prediction_data}")
                        
                        # Keep only last 20 predictions
                        if len(predictions) > 20:
                            predictions = predictions[-20:]
                            
                    except json.JSONDecodeError as e:
                        print(f"❌ Error parsing prediction: {e}")
    except Exception as e:
        print(f"❌ Error monitoring predictions: {e}")

@app.route('/get_predictions')
def get_predictions():
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    print("🚀 Starting E-Waste Classification Dashboard...")
    print("📱 Open http://localhost:5000 in your browser")
    app.run(debug=True, host='127.0.0.1', port=5000)
