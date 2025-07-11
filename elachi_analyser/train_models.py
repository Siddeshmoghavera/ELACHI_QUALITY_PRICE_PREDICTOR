import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import pickle
import joblib

def load_and_prepare_data():
    """Load and prepare the dataset for training"""
    print("Loading dataset...")
    df = pd.read_csv('elaichi_dataset.csv')
    
    # Features for training
    feature_columns = ['Moisture', 'Size', 'Color', 'Aroma', 'Oil_Content']
    X = df[feature_columns]
    
    # Target variables
    y_price = df['Price_per_kg']
    y_quality = df['Quality_Label']
    
    return X, y_price, y_quality, feature_columns

def train_price_prediction_model(X, y_price):
    """Train Random Forest Regressor for price prediction"""
    print("\nTraining Price Prediction Model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_price, test_size=0.2, random_state=42
    )
    
    # Train Random Forest Regressor
    rf_regressor = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    rf_regressor.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_regressor.predict(X_test)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Price Prediction Model Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {np.mean(np.abs(y_test - y_pred)):.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_regressor.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance for Price Prediction:")
    print(feature_importance)
    
    return rf_regressor

def train_quality_classification_model(X, y_quality):
    """Train Random Forest Classifier for quality classification"""
    print("\nTraining Quality Classification Model...")
    
    # Encode quality labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_quality)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Train Random Forest Classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_classifier.predict(X_test)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Quality Classification Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification report
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred_labels))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance for Quality Classification:")
    print(feature_importance)
    
    return rf_classifier, label_encoder

def save_models(price_model, quality_model, label_encoder, feature_columns):
    """Save trained models and encoders"""
    print("\nSaving models...")
    
    # Save models using joblib (recommended for scikit-learn)
    joblib.dump(price_model, 'price_prediction_model.pkl')
    joblib.dump(quality_model, 'quality_classification_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    # Save feature columns for consistency
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    
    print("Models saved successfully!")
    print("Files created:")
    print("- price_prediction_model.pkl")
    print("- quality_classification_model.pkl")
    print("- label_encoder.pkl")
    print("- feature_columns.pkl")

def main():
    """Main training pipeline"""
    try:
        # Load and prepare data
        X, y_price, y_quality, feature_columns = load_and_prepare_data()
        
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {feature_columns}")
        
        # Train price prediction model
        price_model = train_price_prediction_model(X, y_price)
        
        # Train quality classification model
        quality_model, label_encoder = train_quality_classification_model(X, y_quality)
        
        # Save models
        save_models(price_model, quality_model, label_encoder, feature_columns)
        
        print("\n✅ Model training completed successfully!")
        
    except FileNotFoundError:
        print("❌ Error: 'elaichi_dataset.csv' not found!")
        print("Please run the dataset generator first.")
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")

if __name__ == "__main__":
    main()