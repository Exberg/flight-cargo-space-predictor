"""
Data processing pipeline for flight cargo prediction
Handles CSV loading, cleaning, feature engineering, and preprocessing
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


class DataProcessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'luggage_size'  # Default target
        
    def load_data(self, file_path):
        """Load CSV data"""
        df = pd.read_csv(file_path)
        return df
    
    def detect_target_column(self, df):
        """Auto-detect target column (luggage_size or similar)"""
        possible_targets = ['luggage_size', 'luggage_volume', 'cargo_size', 
                          'baggage_size', 'bag_size', 'luggage_weight']
        for col in df.columns:
            if any(target in col.lower() for target in possible_targets):
                self.target_column = col
                return col
        # If not found, use last column as target
        self.target_column = df.columns[-1]
        return self.target_column
    
    def preprocess(self, df, is_training=True):
        """Preprocess data: handle missing values, encode categorical, scale"""
        df = df.copy()
        
        # Detect target if training
        if is_training:
            target_col = self.detect_target_column(df)
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Separate features and target
        if is_training:
            feature_cols = [col for col in df.columns if col != self.target_column]
            X = df[feature_cols].copy()
            y = df[self.target_column].copy()
        else:
            feature_cols = [col for col in df.columns]
            X = df[feature_cols].copy()
            y = None
        
        # Handle missing values
        X = X.fillna(X.median(numeric_only=True))
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'unknown')
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if is_training:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    X[col] = X[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    X[col] = 0  # Default value
        
        # Scale numerical features
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_columns = X.columns.tolist()
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns, index=X.index)
        
        if is_training:
            return X_scaled, y
        else:
            return X_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def save_processor(self, file_path):
        """Save processor state"""
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }, file_path)
    
    def load_processor(self, file_path):
        """Load processor state"""
        data = joblib.load(file_path)
        self.label_encoders = data['label_encoders']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.target_column = data['target_column']



