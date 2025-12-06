"""
SHAP explainability module for feature importance and interactions
"""
import shap
import pandas as pd
import numpy as np
import json


class SHAPExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self, X_background, explainer_type='tree'):
        """Create SHAP explainer"""
        if explainer_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif explainer_type == 'kernel':
            self.explainer = shap.KernelExplainer(self.model.predict, X_background)
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
        return self.explainer
    
    def explain(self, X):
        """Generate SHAP values for predictions"""
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
        
        self.shap_values = self.explainer.shap_values(X)
        return self.shap_values
    
    def get_feature_importance(self, X):
        """Get feature importance scores"""
        if self.shap_values is None:
            self.explain(X)
        
        # Calculate mean absolute SHAP values
        if isinstance(self.shap_values, list):
            shap_values = self.shap_values[0] if len(self.shap_values) > 0 else self.shap_values
        else:
            shap_values = self.shap_values
        
        importance = np.abs(shap_values).mean(0)
        importance_dict = dict(zip(self.feature_names, importance))
        return sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    def get_feature_interactions(self, X, top_n=10):
        """Get top feature interactions"""
        if self.explainer is None:
            self.create_explainer(X)
        
        # Use SHAP interaction values for tree models
        if hasattr(self.explainer, 'shap_interaction_values'):
            try:
                interaction_values = self.explainer.shap_interaction_values(X)
                # Get mean absolute interactions
                interactions = np.abs(interaction_values).mean(0)
                
                interactions_list = []
                for i in range(len(self.feature_names)):
                    for j in range(i+1, len(self.feature_names)):
                        interactions_list.append({
                            'feature1': self.feature_names[i],
                            'feature2': self.feature_names[j],
                            'interaction_strength': float(interactions[i, j])
                        })
                
                # Sort by interaction strength
                interactions_list.sort(key=lambda x: x['interaction_strength'], reverse=True)
                return interactions_list[:top_n]
            except:
                pass
        
        # Fallback: use correlation of SHAP values
        shap_vals = self.explain(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0] if len(shap_vals) > 0 else shap_vals
        
        shap_df = pd.DataFrame(shap_vals, columns=self.feature_names)
        correlations = shap_df.corr().abs()
        
        interactions_list = []
        for i in range(len(self.feature_names)):
            for j in range(i+1, len(self.feature_names)):
                interactions_list.append({
                    'feature1': self.feature_names[i],
                    'feature2': self.feature_names[j],
                    'interaction_strength': float(correlations.iloc[i, j])
                })
        
        interactions_list.sort(key=lambda x: x['interaction_strength'], reverse=True)
        return interactions_list[:top_n]
    
    def get_insights(self, X, y_pred=None):
        """Generate comprehensive insights"""
        insights = {
            'feature_importance': [],
            'feature_interactions': [],
            'predictions': []
        }
        
        # Feature importance
        importance = self.get_feature_importance(X)
        insights['feature_importance'] = [
            {'feature': feat, 'importance': float(imp)} 
            for feat, imp in importance
        ]
        
        # Feature interactions
        interactions = self.get_feature_interactions(X)
        insights['feature_interactions'] = interactions
        
        # Individual predictions with explanations
        if y_pred is not None:
            shap_vals = self.explain(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0] if len(shap_vals) > 0 else shap_vals
            
            for idx in range(min(5, len(X))):  # Limit to 5 examples
                pred_insight = {
                    'prediction': float(y_pred[idx]),
                    'top_contributors': []
                }
                
                # Get top contributing features for this prediction
                contributions = list(zip(self.feature_names, shap_vals[idx]))
                contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                
                for feat, contrib in contributions[:5]:
                    pred_insight['top_contributors'].append({
                        'feature': feat,
                        'contribution': float(contrib)
                    })
                
                insights['predictions'].append(pred_insight)
        
        return insights



