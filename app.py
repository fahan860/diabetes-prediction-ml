from flask import Flask, render_template, request, jsonify
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ajouter le répertoire parent au chemin
sys.path.insert(0, str(Path(__file__).parent))

from src.config import METRICS_PATH, TRAINED_MODEL_PATH, DATA_PATH


app = Flask(__name__)

# Charger les données et modèles au démarrage
def load_data():
    """Charger les métriques et données"""
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)
    
    df = pd.read_csv(DATA_PATH)
    best_model = joblib.load(TRAINED_MODEL_PATH)
    
    return metrics, df, best_model

metrics, df, best_model = load_data()

# Colonnes des features (basées sur le dataset Pima Indians)
FEATURE_COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

@app.route('/')
def index():
    """Page d'accueil avec résultats des modèles"""
    # Préparer les données pour le template
    models_comparison = []
    for model_name, model_metrics in metrics.items():
        models_comparison.append({
            'name': model_name,
            'accuracy': f"{model_metrics['accuracy']:.2%}",
            'precision': f"{model_metrics['precision']:.2%}",
            'recall': f"{model_metrics['recall']:.2%}",
            'f1_score': f"{model_metrics['f1']:.2%}",
            'accuracy_raw': model_metrics['accuracy']
        })
    
    # Trier par accuracy
    models_comparison.sort(key=lambda x: x['accuracy_raw'], reverse=True)
    
    return render_template('index.html', models=models_comparison)

@app.route('/compare')
def compare():
    """Page de comparaison détaillée des modèles"""
    # Préparer les données pour la comparaison
    comparison_data = {
        'accuracy': {},
        'precision': {},
        'recall': {},
        'f1_score': {}
    }
    
    for model_name, model_metrics in metrics.items():
        comparison_data['accuracy'][model_name] = model_metrics['accuracy']
        comparison_data['precision'][model_name] = model_metrics['precision']
        comparison_data['recall'][model_name] = model_metrics['recall']
        comparison_data['f1_score'][model_name] = model_metrics['f1']
    
    return render_template('compare.html', data=comparison_data)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Page et API de prédiction"""
    if request.method == 'GET':
        # Obtenir les statistiques des features
        feature_stats = {}
        for col in FEATURE_COLUMNS:
            feature_stats[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
        
        return render_template('predict.html', feature_stats=feature_stats, features=FEATURE_COLUMNS)
    
    elif request.method == 'POST':
        try:
            # Récupérer les données du formulaire
            data = request.json
            
            # Préparer les features
            features = []
            for col in FEATURE_COLUMNS:
                try:
                    value = float(data.get(col, 0))
                except (ValueError, TypeError):
                    return jsonify({'error': f'Invalid value for {col}'}), 400
                features.append(value)
            
            # Faire la prédiction
            prediction = best_model.predict([features])[0]
            probability = best_model.predict_proba([features])[0]
            
            result = {
                'prediction': int(prediction),
                'probability_negative': float(probability[0]),
                'probability_positive': float(probability[1]),
                'risk_level': 'High Risk' if prediction == 1 else 'Low Risk'
            }
            
            return jsonify(result)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 400

@app.route('/api/metrics')
def api_metrics():
    """API pour obtenir toutes les métriques"""
    return jsonify(metrics)

@app.route('/about')
def about():
    """Page à propos"""
    return render_template('about.html')

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
