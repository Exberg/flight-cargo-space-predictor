"""
ML Model Trainer for flight cargo prediction
Implements XGBoost, LightGBM, and CatBoost models
"""
import pandas as pd
import numpy as np
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    XGBOOST_AVAILABLE = False
    print(f"Warning: XGBoost not available: {type(e).__name__}")
    print("Install libomp: brew install libomp (or continue without XGBoost)")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    LIGHTGBM_AVAILABLE = False
    print(f"Warning: LightGBM not available: {type(e).__name__}")
    print("Install libomp: brew install libomp (or continue without LightGBM)")

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import json


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.model_scores = {}
        
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train XGBoost model"""
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            **kwargs
        }
        model = XGBRegressor(**params)
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train LightGBM model"""
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': -1,
            **kwargs
        }
        model = LGBMRegressor(**params)
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)
        return model
    
    def train_catboost(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train CatBoost model"""
        params = {
            'iterations': 100,
            'depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': False,
            **kwargs
        }
        model = CatBoostRegressor(**params)
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        else:
            model.fit(X_train, y_train, verbose=False)
        return model
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train all available models"""
        if XGBOOST_AVAILABLE:
            print("Training XGBoost...")
            try:
                self.models['xgboost'] = self.train_xgboost(X_train, y_train, X_val, y_val)
            except Exception as e:
                print(f"Warning: XGBoost training failed: {e}")
        else:
            print("Skipping XGBoost (not available)")
        
        if LIGHTGBM_AVAILABLE:
            print("Training LightGBM...")
            try:
                self.models['lightgbm'] = self.train_lightgbm(X_train, y_train, X_val, y_val)
            except Exception as e:
                print(f"Warning: LightGBM training failed: {e}")
        else:
            print("Skipping LightGBM (not available)")
        
        print("Training CatBoost...")
        self.models['catboost'] = self.train_catboost(X_train, y_train, X_val, y_val)
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and return metrics"""
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            results[name] = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            self.model_scores[name] = results[name]
        return results
    
    def get_best_model(self):
        """Get the best performing model based on R2 score"""
        if not self.model_scores:
            return None, None
        best_name = max(self.model_scores.keys(), 
                       key=lambda x: self.model_scores[x]['r2'])
        return best_name, self.models[best_name]
    
    def predict(self, X, model_name=None):
        """Make predictions using specified model or best model"""
        if model_name and model_name in self.models:
            model = self.models[model_name]
        else:
            _, model = self.get_best_model()
            if model is None:
                raise ValueError("No trained models available")
        
        return model.predict(X)
    
    def save_models(self, directory='models'):
        """Save all trained models"""
        os.makedirs(directory, exist_ok=True)
        for name, model in self.models.items():
            model_path = os.path.join(directory, f'{name}_model.pkl')
            joblib.dump(model, model_path)
        
        # Save scores
        scores_path = os.path.join(directory, 'model_scores.json')
        with open(scores_path, 'w') as f:
            json.dump(self.model_scores, f, indent=2)
    
    def load_models(self, directory='models'):
        """Load all trained models"""
        for name in ['xgboost', 'lightgbm', 'catboost']:
            model_path = os.path.join(directory, f'{name}_model.pkl')
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
        
        # Load scores
        scores_path = os.path.join(directory, 'model_scores.json')
        if os.path.exists(scores_path):
            with open(scores_path, 'r') as f:
                self.model_scores = json.load(f)



