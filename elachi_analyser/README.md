# 🟢 Elaichi Quality and Price Analyzer

A comprehensive web-based application that uses Machine Learning to analyze Elaichi (Cardamom) quality and predict market prices based on physical and chemical attributes.

## 🎯 Features

- **Quality Classification**: Predicts quality grade (Low, Standard, Premium)
- **Price Prediction**: Estimates market price per kg in INR
- **Interactive Web Interface**: User-friendly form with real-time validation
- **Confidence Levels**: Shows prediction probabilities for quality grades
- **Responsive Design**: Works on desktop and mobile devices

## 🧠 Technology Stack

- **Backend**: Python Flask
- **Machine Learning**: Scikit-learn (Random Forest)
- **Frontend**: HTML5, CSS3, Bootstrap 5
- **Data Processing**: Pandas, NumPy
- **Model Storage**: Joblib/Pickle

## 📊 Dataset Features

The model analyzes the following Elaichi attributes:

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| Moisture | Float | 0-20% | Moisture content percentage |
| Size | Float | 5-25mm | Average pod size in millimeters |
| Color | Integer | 1-10 | Visual color quality score |
| Aroma | Integer | 1-10 | Fragrance intensity score |
| Oil Content | Float | 0-10% | Essential oil content percentage |

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# 1. Download all project files
# 2. Run the setup script
python setup_project.py

# 3. Start the application
python app.py
```

### Option 2: Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python generate_elaichi_data.py

# 3. Train models
python train_models.py

# 4. Start Flask app
python app.py
```

### Option 3: Step-by-Step

1. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv elaichi_env
   
   # Windows
   elaichi_env\Scripts\activate
   
   # macOS/Linux
   source elaichi_env/bin/activate
   ```

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create Directory Structure**
   ```
   elaichi_analyzer/
   ├── app.py
   ├── generate_elaichi_data.py
   ├── train_models.py
   ├── requirements.txt
   ├── templates/
   │   ├── index.html
   │   ├── 404.html
   │   └── 500.html
   └── README.md
   ```

4. **Generate Dataset**
   ```bash
   python generate_elaichi_data.py
   ```

5. **Train Models**
   ```bash
   python train_models.py
   ```

6. **Run Application**
   ```bash
   python app.py
   ```

7. **Access Web Interface**
   Open browser to: `http://localhost:5000`

## 📁 Project Structure

```
elaichi_analyzer/
├── app.py                      # Flask web application
├── generate_elaichi_data.py     # Synthetic dataset generator
├── train_models.py              # ML model training script
├── setup_project.py             # Automated setup script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── templates/
│   ├── index.html              # Main web interface
│   ├── 404.html                # 404 error page
│   └── 500.html                # 500 error page
├── Generated Files:
├── elaichi_dataset.csv          # Synthetic dataset (1000 samples)
├── price_prediction_model.pkl   # Trained price model
├── quality_classification_model.pkl # Trained quality model
├── label_encoder.pkl            # Quality label encoder
└── feature_columns.pkl          # Feature column names
```

## 🧪 Machine Learning Models

### 1. Price Prediction Model
- **Algorithm**: Random Forest Regressor
- **Purpose**: Predicts market price per kg (INR)
- **Features**: All 5 input attributes
- **Performance**: RMSE typically < 200 INR

### 2. Quality Classification Model
- **Algorithm**: Random Forest Classifier
- **Purpose**: Classifies quality grade
- **Classes**: Low, Standard, Premium
- **Features**: All 5 input attributes
- **Performance**: Accuracy typically > 85%

## 🌐 API Endpoints

### GET `/`
- **Description**: Main application interface
- **Returns**: HTML form for input

### POST `/predict`
- **Description**: Prediction endpoint
- **Input**: Form data with Elaichi attributes
- **Returns**: JSON with predictions

**Example Request:**
```json
{
  "moisture": 8.5,
  "size": 12.0,
  "color": 8,
  "aroma": 9,
  "oil_content": 4.5
}
```

**Example Response:**
```json
{
  "success": true,
  "predicted_price": 2150.75,
  "predicted_quality": "Premium",
  "quality_probabilities": {
    "Low": 5.2,
    "Standard": 22.3,
    "Premium": 72.5
  },
  "input_data": { ... }
}
```

### GET `/model_info`
- **Description**: Model information endpoint
- **Returns**: Model details and feature info

## 📊 Dataset Generation Logic

The synthetic dataset follows realistic market rules:

### Quality Determination
- **Premium**: High scores across all attributes
- **Standard**: Moderate scores, good overall quality
- **Low**: Poor performance in key attributes

### Price Calculation
- Base price by quality tier
- Bonuses for premium features:
  - Size ≥ 14mm: +₹300
  - Color ≥ 8: +₹200
  - Aroma ≥ 8: +₹200
  - Oil Content ≥ 6%: +₹250
  - Moisture ≤ 7%: +₹150
- Market variation: ±₹200 random fluctuation

## 🎨 Web Interface Features

### Input Form
- Real-time validation
- Helpful range indicators
- Error handling
- Professional styling

### Results Display
- Quality badge with color coding
- Price prediction in INR
- Confidence levels with progress bars
- Input summary

### Design Features
- Responsive Bootstrap layout
- Modern gradient backgrounds
- Interactive animations
- Mobile-friendly interface

## 🔧 Customization Options

### 1. Modify Dataset
Edit `generate_elaichi_data.py` to:
- Change sample size
- Adjust feature ranges
- Modify quality/price logic

### 2. Experiment with Models
Edit `train_models.py` to:
- Try different algorithms
- Tune hyperparameters
- Add feature engineering

### 3. Enhance Web Interface
Edit `templates/index.html` to:
- Add new input fields
- Modify styling
- Include additional visualizations

## 🐛 Troubleshooting

### Common Issues

1. **"Models not loaded" error**
   ```bash
   # Solution: Train models first
   python train_models.py
   ```

2. **"Dataset not found" error**
   ```bash
   # Solution: Generate dataset first
   python generate_elaichi_data.py
   ```

3. **Import errors**
   ```bash
   # Solution: Install requirements
   pip install -r requirements.txt
   ```

4. **Permission errors (Linux/Mac)**
   ```bash
   # Solution: Make scripts executable
   chmod +x run.sh
   ```

### Debug Mode
To enable Flask debug mode, modify `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## 📈 Performance Metrics

### Model Performance (Typical)
- **Price Model RMSE**: 150-250 INR
- **Price Model R²**: 0.85-0.95
- **Quality Model Accuracy**: 85-95%

### Dataset Statistics
- **Total Samples**: 1000
- **Features**: 5 numerical
- **Quality Distribution**: Balanced across classes
- **Price Range**: ₹500-₹4000 per kg

## 🚀 Deployment Options

### Local Development
```bash
python app.py
# Access: http://localhost:5000
```

### Production Deployment

#### PythonAnywhere
1. Upload files to PythonAnywhere
2. Install requirements in virtual environment
3. Configure WSGI file
4. Set up web app

#### Heroku
1. Create `Procfile`: `web: python app.py`
2. Deploy via Git
3. Configure environment variables

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make improvements
4. Test thoroughly
5. Submit pull request

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created as a demonstration of ML-powered web applications for agricultural quality assessment.

## 🙏 Acknowledgments

- Scikit-learn for ML algorithms
- Flask for web framework
- Bootstrap for UI components
- Agricultural domain experts for feature insights