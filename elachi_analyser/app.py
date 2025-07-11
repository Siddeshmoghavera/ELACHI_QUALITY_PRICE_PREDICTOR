from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Global variables to store models
price_model = None
quality_model = None
label_encoder = None
feature_columns = None

def load_models():
    """Load trained models and encoders"""
    global price_model, quality_model, label_encoder, feature_columns
    
    try:
        # Load models
        price_model = joblib.load('price_prediction_model.pkl')
        quality_model = joblib.load('quality_classification_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        
        # Load feature columns
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        print("✅ Models loaded successfully!")
        return True
        
    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        print("Please train the models first by running train_models.py")
        return False
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def validate_input(data):
    """Validate input data"""
    errors = []
    
    try:
        moisture = float(data.get('moisture', 0))
        if not (0 <= moisture <= 20):
            errors.append("Moisture must be between 0-20%")
    except ValueError:
        errors.append("Moisture must be a valid number")
    
    try:
        size = float(data.get('size', 0))
        if not (5 <= size <= 25):
            errors.append("Size must be between 5-25 mm")
    except ValueError:
        errors.append("Size must be a valid number")
    
    try:
        color = int(data.get('color', 0))
        if not (1 <= color <= 10):
            errors.append("Color score must be between 1-10")
    except ValueError:
        errors.append("Color must be a valid integer")
    
    try:
        aroma = int(data.get('aroma', 0))
        if not (1 <= aroma <= 10):
            errors.append("Aroma score must be between 1-10")
    except ValueError:
        errors.append("Aroma must be a valid integer")
    
    try:
        oil_content = float(data.get('oil_content', 0))
        if not (0 <= oil_content <= 10):
            errors.append("Oil content must be between 0-10%")
    except ValueError:
        errors.append("Oil content must be a valid number")
    
    return errors

@app.route('/')
def index():
    """Render the main input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Check if models are loaded
        if price_model is None or quality_model is None:
            return jsonify({
                'success': False,
                'error': 'Models not loaded. Please check server logs.'
            })
        
        # Get form data
        form_data = request.form.to_dict()
        
        # Validate input
        validation_errors = validate_input(form_data)
        if validation_errors:
            return jsonify({
                'success': False,
                'error': 'Validation errors: ' + '; '.join(validation_errors)
            })
        
        # Prepare input features
        input_features = [
            float(form_data['moisture']),
            float(form_data['size']),
            int(form_data['color']),
            int(form_data['aroma']),
            float(form_data['oil_content'])
        ]
        
        # Convert to numpy array for prediction
        X_input = np.array([input_features])
        
        # Make predictions
        predicted_price = price_model.predict(X_input)[0]
        predicted_quality_encoded = quality_model.predict(X_input)[0]
        predicted_quality = label_encoder.inverse_transform([predicted_quality_encoded])[0]
        
        # Get prediction probabilities for quality
        quality_probabilities = quality_model.predict_proba(X_input)[0]
        quality_classes = label_encoder.inverse_transform(range(len(quality_probabilities)))
        
        # Create probability dictionary
        quality_proba_dict = {}
        for cls, prob in zip(quality_classes, quality_probabilities):
            quality_proba_dict[cls] = round(prob * 100, 2)
        
        # Round predicted price
        predicted_price = round(predicted_price, 2)
        
        return jsonify({
            'success': True,
            'predicted_price': predicted_price,
            'predicted_quality': predicted_quality,
            'quality_probabilities': quality_proba_dict,
            'input_data': {
                'moisture': float(form_data['moisture']),
                'size': float(form_data['size']),
                'color': int(form_data['color']),
                'aroma': int(form_data['aroma']),
                'oil_content': float(form_data['oil_content'])
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        })

@app.route('/model_info')
def model_info():
    """Provide information about the loaded models"""
    if price_model is None or quality_model is None:
        return jsonify({
            'success': False,
            'error': 'Models not loaded'
        })
    
    return jsonify({
        'success': True,
        'feature_columns': feature_columns,
        'price_model_type': str(type(price_model).__name__),
        'quality_model_type': str(type(quality_model).__name__),
        'quality_classes': label_encoder.classes_.tolist()
    })

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("🌱 Starting Elaichi Quality and Price Analyzer...")
    
    # Load models on startup
    models_loaded = load_models()
    
    if not models_loaded:
        print("⚠️  Warning: Models not loaded. Some features may not work.")
        print("Please run the following commands first:")
        print("1. python generate_elaichi_data.py")
        print("2. python train_models.py")
    else:
        print("🚀 All models loaded successfully!")
    
    print("📡 Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)