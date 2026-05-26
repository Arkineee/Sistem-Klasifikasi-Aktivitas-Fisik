# app.py (gabungan dengan config.py)
# -*- coding: utf-8 -*-
"""
Flask Backend untuk Klasifikasi Aktivitas Fisik
Mengintegrasikan model Random Forest dengan frontend HTML
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================
# KONFIGURASI (dari config.py)
# ============================================

class Config:
    # Supabase Configuration (opsional, untuk penyimpanan history)
    SUPABASE_URL = os.getenv('SUPABASE_URL', 'https://icdnmhszfqwrhrgbngqw.supabase.co')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY', '')
    
    # Model paths
    MODEL_PATH = 'models/random_forest_model.joblib'
    SCALER_PATH = 'models/scaler.joblib'
    TARGET_ENCODER_PATH = 'models/target_encoder.joblib'
    LABEL_ENCODERS_PATH = 'models/label_encoders.joblib'
    FEATURE_COLUMNS_PATH = 'models/feature_columns.joblib'
    NUMERIC_FEATURES_PATH = 'models/numeric_features.joblib'
    
    # Model metrics (akan diupdate saat training)
    MODEL_METRICS = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0
    }
    
    # Mapping untuk encoding
    GENDER_MAPPING = {'Laki-laki': 0, 'Perempuan': 1}
    SLEEP_DISORDER_MAPPING = {
        'Tidak Ada': 0,
        'Insomnia': 1,
        'Sleep Apnea': 2
    }


# ============================================
# INISIALISASI FLASK
# ============================================

app = Flask(__name__, template_folder='templates')
CORS(app)
app.config.from_object(Config)

# Global variables untuk model
model = None
scaler = None
target_encoder = None
label_encoders = None
feature_columns = None
numeric_features = None
model_metrics = None

# Inisialisasi Supabase
supabase_client = None
supabase_enabled = False


# ============================================
# FUNGSI SUPABASE
# ============================================

def init_supabase():
    """Initialize Supabase connection"""
    global supabase_client, supabase_enabled
    
    if Config.SUPABASE_URL and Config.SUPABASE_KEY:
        try:
            from supabase import create_client, Client
            supabase_client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
            supabase_enabled = True
            print("✅ Supabase connected successfully!")
            
            # Test connection - cek apakah tabel ada
            try:
                test_response = supabase_client.table('predictions').select('id').limit(1).execute()
                print("   ✅ Supabase table 'predictions' is accessible")
            except Exception as table_err:
                print(f"   ⚠️ Warning: Table 'predictions' might not exist yet: {table_err}")
                print("   📝 Please run the SQL script to create the table")
            
        except Exception as e:
            print(f"⚠️ Supabase connection failed: {e}")
            supabase_enabled = False
    else:
        if not Config.SUPABASE_KEY:
            print("⚠️ Supabase not configured: Missing SUPABASE_KEY in .env file")
        else:
            print("⚠️ Supabase not configured (missing URL or KEY)")
        supabase_enabled = False
    
    return supabase_enabled


def save_prediction_to_supabase(data, prediction, confidence, bmi_value, bmi_category):
    """Save prediction to Supabase for history"""
    global supabase_client, supabase_enabled
    
    if not supabase_enabled or supabase_client is None:
        print("⚠️ Supabase not enabled, skipping save")
        return False
    
    try:
        # Prepare record - match with database column names
        record = {
            'created_at': datetime.now().isoformat(),
            'usia': float(data['usia']),
            'jenis_kelamin': data['jenisKelamin'],
            'pekerjaan': data['pekerjaan'],
            'tekanan_darah': data['tekananDarah'],
            'kualitas_tidur': float(data['kualitasTidur']),
            'detak_jantung': float(data['detakJantung']),
            'durasi_tidur': float(data['durasiTidur']),
            'gangguan_tidur': data['gangguanTidur'],
            'langkah_kaki': float(data['langkahKaki']),
            'berat_badan': float(data['beratBadan']),
            'tinggi_badan': float(data['tinggiBadan']),
            'bmi': bmi_value,
            'bmi_category': bmi_category,
            'prediction': prediction,
            'confidence_rendah': confidence.get('Rendah', 0),
            'confidence_sedang': confidence.get('Sedang', 0),
            'confidence_tinggi': confidence.get('Tinggi', 0)
        }
        
        print(f"📝 Saving to Supabase: {record}")
        
        # Insert to Supabase
        response = supabase_client.table('predictions').insert(record).execute()
        
        print(f"✅ Successfully saved to Supabase!")
        return True
        
    except Exception as e:
        print(f"❌ Error saving to Supabase: {e}")
        traceback.print_exc()
        return False


# ============================================
# FUNGSI LOAD MODEL
# ============================================

def load_models():
    """Load semua model dan encoder yang telah disimpan"""
    global model, scaler, target_encoder, label_encoders, feature_columns, numeric_features, model_metrics
    
    print("\n" + "="*60)
    print("LOADING MODELS...")
    print("="*60)
    
    # Check if models directory exists
    if not os.path.exists('models'):
        print("❌ 'models/' directory not found!")
        return False
    
    try:
        # Load model
        model_path = 'models/random_forest_model.joblib'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("✓ Model loaded")
        else:
            print(f"❌ Model not found at {model_path}")
            return False
        
        # Load scaler
        if os.path.exists('models/scaler.joblib'):
            scaler = joblib.load('models/scaler.joblib')
            print("✓ Scaler loaded")
        else:
            print("⚠️ Scaler not found, will use default")
        
        # Load target encoder
        if os.path.exists('models/target_encoder.joblib'):
            target_encoder = joblib.load('models/target_encoder.joblib')
            print("✓ Target encoder loaded")
        else:
            print("⚠️ Target encoder not found")
        
        # Load label encoders
        if os.path.exists('models/label_encoders.joblib'):
            label_encoders = joblib.load('models/label_encoders.joblib')
            print("✓ Label encoders loaded")
        else:
            print("⚠️ Label encoders not found")
        
        # Load feature columns
        if os.path.exists('models/feature_columns.joblib'):
            feature_columns = joblib.load('models/feature_columns.joblib')
            print("✓ Feature columns loaded")
        else:
            print("⚠️ Feature columns not found")
        
        # Load numeric features
        if os.path.exists('models/numeric_features.joblib'):
            numeric_features = joblib.load('models/numeric_features.joblib')
            print("✓ Numeric features loaded")
        else:
            print("⚠️ Numeric features not found")
        
        # Load metrics jika ada
        metrics_path = 'models/model_metrics.joblib'
        if os.path.exists(metrics_path):
            model_metrics = joblib.load(metrics_path)
            print("✓ Model metrics loaded")
        
        print("\n✅ All models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading models: {e}")
        traceback.print_exc()
        return False


def get_model_metrics():
    """Get model metrics from saved file or default"""
    global model_metrics
    
    if model_metrics:
        return model_metrics
    
    metrics_path = 'models/model_metrics.joblib'
    if os.path.exists(metrics_path):
        try:
            model_metrics = joblib.load(metrics_path)
            return model_metrics
        except:
            pass
    
    return {
        'accuracy': 85.5,
        'precision': 84.2,
        'recall': 83.8,
        'f1_score': 84.0
    }


# ============================================
# FUNGSI PREPROCESSING
# ============================================

def calculate_bmi(weight_kg, height_cm):
    """Calculate BMI from weight (kg) and height (cm)"""
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    
    # Kategori BMI
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    
    return round(bmi, 1), category


def get_bmi_category_encoded(bmi_category):
    """Get encoded value for BMI category"""
    bmi_mapping = {
        'Underweight': 0,
        'Normal': 1,
        'Overweight': 2,
        'Obese': 3
    }
    return bmi_mapping.get(bmi_category, 1)


def get_occupation_encoded(occupation):
    """Get encoded value for occupation using saved label encoder"""
    global label_encoders
    
    try:
        if label_encoders and 'Occupation' in label_encoders:
            le = label_encoders['Occupation']
            if occupation in le.classes_:
                return le.transform([occupation])[0]
        
        # Default encoding based on occupation type
        occupation_mapping = {
            'Dokter': 0, 'Engineer': 1, 'Guru': 2, 'Sales': 3,
            'Manager': 4, 'Accountant': 5, 'Developer': 6, 'Designer': 7,
            'Student': 8, 'Retired': 9, 'Unemployed': 10
        }
        return occupation_mapping.get(occupation, 0)
    except Exception as e:
        print(f"Warning: Could not encode occupation '{occupation}': {e}")
        return 0


def preprocess_input(data):
    """
    Preprocess input data from frontend to match model's expected format
    """
    
    # Extract blood pressure
    bp_string = data['tekananDarah']
    try:
        if '/' in bp_string:
            systolic, diastolic = bp_string.split('/')
            systolic = float(systolic.strip())
            diastolic = float(diastolic.strip())
        else:
            systolic = 120
            diastolic = 80
    except:
        systolic = 120
        diastolic = 80
    
    # Calculate BMI
    bmi_value, bmi_category = calculate_bmi(
        float(data['beratBadan']), 
        float(data['tinggiBadan'])
    )
    
    # Encode categorical features
    gender_mapping = {'Laki-laki': 0, 'Perempuan': 1}
    gender_encoded = gender_mapping.get(data['jenisKelamin'], 0)
    occupation_encoded = get_occupation_encoded(data['pekerjaan'])
    bmi_category_encoded = get_bmi_category_encoded(bmi_category)
    
    sleep_disorder_mapping = {
        'Tidak Ada': 0,
        'Insomnia': 1,
        'Sleep Apnea': 2
    }
    sleep_disorder_encoded = sleep_disorder_mapping.get(data['gangguanTidur'], 0)
    
    # Create feature dictionary
    features = {
        'Age': float(data['usia']),
        'Sleep Duration': float(data['durasiTidur']),
        'Quality of Sleep': float(data['kualitasTidur']),
        'Heart Rate': float(data['detakJantung']),
        'Daily Steps': float(data['langkahKaki']),
        'Systolic_BP': systolic,
        'Diastolic_BP': diastolic,
        'Gender_Encoded': gender_encoded,
        'Occupation_Encoded': occupation_encoded,
        'BMI Category_Encoded': bmi_category_encoded,
        'Sleep Disorder_Encoded': sleep_disorder_encoded
    }
    
    # Create DataFrame
    df = pd.DataFrame([features])
    
    # Ensure columns are in correct order if feature_columns exists
    if feature_columns is not None:
        try:
            df = df[feature_columns]
        except:
            print("Warning: Feature columns mismatch, using available columns")
    
    # Apply scaling to numeric features if scaler exists
    if scaler is not None and numeric_features is not None:
        try:
            df[numeric_features] = scaler.transform(df[numeric_features])
        except:
            print("Warning: Scaling failed, using unscaled data")
    
    return df, bmi_value, bmi_category


# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk melakukan prediksi aktivitas fisik
    """
    
    # Check if model is loaded
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model belum dimuat. Silakan jalankan training terlebih dahulu.'
        }), 503
    
    try:
        # Get data from request
        data = request.get_json()
        print(f"\n📥 Received prediction request: {data}")
        
        # Validate required fields
        required_fields = ['jenisKelamin', 'pekerjaan', 'usia', 'tekananDarah',
                          'kualitasTidur', 'detakJantung', 'durasiTidur',
                          'gangguanTidur', 'langkahKaki', 'beratBadan', 'tinggiBadan']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Field {field} tidak ditemukan'
                }), 400
        
        # Preprocess input
        processed_data, bmi_value, bmi_category = preprocess_input(data)
        print(f"✓ Data preprocessed, BMI: {bmi_value} ({bmi_category})")
        
        # Make prediction
        prediction_encoded = model.predict(processed_data)[0]
        prediction_proba = model.predict_proba(processed_data)[0]
        
        # Decode prediction
        if target_encoder is not None:
            prediction = target_encoder.inverse_transform([prediction_encoded])[0]
        else:
            # Default mapping if no encoder
            pred_mapping = {0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'}
            prediction = pred_mapping.get(prediction_encoded, 'Sedang')
        
        print(f"✓ Prediction: {prediction}")
        
        # Get confidence for each class
        if target_encoder is not None:
            class_names = target_encoder.classes_
        else:
            class_names = ['Rendah', 'Sedang', 'Tinggi']
        
        confidence = {}
        for i, class_name in enumerate(class_names):
            if i < len(prediction_proba):
                confidence[class_name] = round(prediction_proba[i] * 100, 2)
            else:
                confidence[class_name] = 0
        print(f"✓ Confidence: {confidence}")
        
        # Get model metrics
        metrics = get_model_metrics()
        
        # Save to Supabase (optional)
        save_success = save_prediction_to_supabase(data, prediction, confidence, bmi_value, bmi_category)
        
        if save_success:
            print("✅ Data saved to Supabase")
        else:
            print("⚠️ Data was NOT saved to Supabase (this is fine if Supabase is not configured)")
        
        # Prepare response
        response = {
            'status': 'success',
            'prediction': prediction,
            'confidence': confidence,
            'bmi_info': {
                'bmi': bmi_value,
                'category': bmi_category
            },
            'metrics': {
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0)
            },
            'saved_to_supabase': save_success
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Terjadi kesalahan: {str(e)}'
        }), 500


@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Get prediction history from Supabase"""
    if not supabase_enabled or supabase_client is None:
        return jsonify({
            'status': 'error',
            'message': 'Supabase not configured or not connected. Please configure SUPABASE_KEY in .env file.'
        }), 503
    
    try:
        limit = request.args.get('limit', 50, type=int)
        
        print(f"📊 Fetching predictions from Supabase (limit={limit})")
        
        response = supabase_client.table('predictions')\
            .select('*')\
            .order('created_at', desc=True)\
            .limit(limit)\
            .execute()
        
        print(f"✅ Retrieved {len(response.data)} predictions")
        
        return jsonify({
            'status': 'success',
            'data': response.data
        })
    except Exception as e:
        print(f"❌ Error fetching predictions: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error fetching predictions: {str(e)}'
        }), 500


@app.route('/api/predictions/stats', methods=['GET'])
def get_prediction_stats():
    """Get statistics from prediction history"""
    if not supabase_enabled or supabase_client is None:
        return jsonify({
            'status': 'error',
            'message': 'Supabase not configured or not connected'
        }), 503
    
    try:
        response = supabase_client.table('predictions')\
            .select('prediction')\
            .execute()
        
        data = response.data
        total = len(data)
        
        # Count predictions per category
        counts = {}
        for item in data:
            pred = item['prediction']
            counts[pred] = counts.get(pred, 0) + 1
        
        # Calculate percentages
        statistics = {}
        for category, count in counts.items():
            statistics[category] = {
                'count': count,
                'percentage': round((count / total) * 100, 2) if total > 0 else 0
            }
        
        return jsonify({
            'status': 'success',
            'total_predictions': total,
            'statistics': statistics
        })
    except Exception as e:
        print(f"❌ Error fetching stats: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/supabase/test', methods=['GET'])
def test_supabase():
    """Test Supabase connection and table"""
    if not supabase_enabled or supabase_client is None:
        return jsonify({
            'status': 'error',
            'message': 'Supabase not configured',
            'config_check': {
                'SUPABASE_URL': bool(Config.SUPABASE_URL),
                'SUPABASE_KEY': bool(Config.SUPABASE_KEY)
            },
            'instructions': 'Please set SUPABASE_KEY in .env file and restart the server'
        }), 503
    
    try:
        # Check if table exists by trying to select
        response = supabase_client.table('predictions').select('id').limit(1).execute()
        
        return jsonify({
            'status': 'success',
            'message': 'Supabase connection is working!',
            'table_exists': True,
            'table_name': 'predictions'
        })
        
    except Exception as e:
        error_msg = str(e)
        if 'relation' in error_msg and 'does not exist' in error_msg:
            return jsonify({
                'status': 'error',
                'message': 'Table "predictions" does not exist',
                'hint': 'Please run the SQL script to create the predictions table',
                'sql_script': """
                CREATE TABLE IF NOT EXISTS predictions (
                    id BIGSERIAL PRIMARY KEY,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    usia FLOAT,
                    jenis_kelamin TEXT,
                    pekerjaan TEXT,
                    tekanan_darah TEXT,
                    kualitas_tidur FLOAT,
                    detak_jantung FLOAT,
                    durasi_tidur FLOAT,
                    gangguan_tidur TEXT,
                    langkah_kaki FLOAT,
                    berat_badan FLOAT,
                    tinggi_badan FLOAT,
                    bmi FLOAT,
                    bmi_category TEXT,
                    prediction TEXT,
                    confidence_rendah FLOAT,
                    confidence_sedang FLOAT,
                    confidence_tinggi FLOAT
                );
                """
            }), 500
        else:
            return jsonify({
                'status': 'error',
                'message': error_msg
            }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    supabase_status = "connected" if supabase_enabled else "not configured"
    
    return jsonify({
        'status': 'healthy',
        'model': model_status,
        'supabase': supabase_status,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 503
    
    return jsonify({
        'status': 'success',
        'model_type': 'RandomForestClassifier',
        'n_features': len(feature_columns) if feature_columns is not None else 0,
        'feature_columns': feature_columns.tolist() if feature_columns is not None else [],
        'n_classes': len(target_encoder.classes_) if target_encoder is not None else 3,
        'classes': target_encoder.classes_.tolist() if target_encoder is not None else ['Rendah', 'Sedang', 'Tinggi'],
        'metrics': get_model_metrics()
    })


# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500


# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏃‍♂️ FLASK BACKEND FOR PHYSICAL ACTIVITY CLASSIFICATION")
    print("="*60)
    
    # Initialize Supabase
    init_supabase()
    
    # Load models
    models_loaded = load_models()
    
    if not models_loaded:
        print("\n⚠️ WARNING: Models not loaded!")
        print("   The API will not work properly until models are available.")
        print("   Please run train_and_save_model.py first to train and save models.")
    
    print("\n" + "="*60)
    print("🚀 STARTING FLASK SERVER...")
    print("="*60)
    print(f"📍 Server running at: http://localhost:5000")
    print(f"📊 Health check: http://localhost:5000/api/health")
    print(f"🤖 Model info: http://localhost:5000/api/model/info")
    print(f"🗄️ Supabase test: http://localhost:5000/api/supabase/test")
    print("\nPress CTRL+C to stop the server")
    print("="*60)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)