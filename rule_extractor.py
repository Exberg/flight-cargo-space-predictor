"""
Decision Tree Rule Extractor for generating human-readable rules
and conditional probabilities
"""
from sklearn.tree import DecisionTreeRegressor, export_text
import pandas as pd
import numpy as np
import json


class RuleExtractor:
    def __init__(self, max_depth=5, min_samples_split=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.feature_names = None
        
    def train(self, X, y, feature_names):
        """Train decision tree for rule extraction"""
        self.feature_names = feature_names
        self.tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=42
        )
        self.tree.fit(X, y)
        return self.tree
    
    def extract_rules(self, X, y):
        """Extract rules from decision tree"""
        if self.tree is None:
            raise ValueError("Tree not trained. Call train() first.")
        
        rules = []
        tree = self.tree.tree_
        
        def recurse(node, depth, parent_conditions=[]):
            if tree.children_left[node] == tree.children_right[node]:  # Leaf node
                samples = tree.n_node_samples[node]
                value = tree.value[node][0][0]
                
                rule = {
                    'conditions': parent_conditions.copy(),
                    'predicted_value': float(value),
                    'samples': int(samples),
                    'percentage': float(samples / len(X) * 100)
                }
                rules.append(rule)
            else:
                feature = self.feature_names[tree.feature[node]]
                threshold = tree.threshold[node]
                
                # Left child (<= threshold)
                left_conditions = parent_conditions + [f"{feature} <= {threshold:.2f}"]
                recurse(tree.children_left[node], depth + 1, left_conditions)
                
                # Right child (> threshold)
                right_conditions = parent_conditions + [f"{feature} > {threshold:.2f}"]
                recurse(tree.children_right[node], depth + 1, right_conditions)
        
        recurse(0, 0)
        return rules
    
    def get_conditional_probabilities(self, X, y, feature_ranges):
        """
        Get conditional probabilities for luggage size ranges
        given feature value ranges
        
        feature_ranges: dict like {'feature1': (min, max), 'feature2': (min, max)}
        """
        if self.tree is None:
            raise ValueError("Tree not trained. Call train() first.")
        
        # Filter data based on conditions
        mask = pd.Series([True] * len(X))
        for feature, (min_val, max_val) in feature_ranges.items():
            if feature in self.feature_names:
                idx = self.feature_names.index(feature)
                feature_values = X[:, idx] if isinstance(X, np.ndarray) else X[feature].values
                mask = mask & (feature_values >= min_val) & (feature_values <= max_val)
        
        filtered_y = y[mask] if isinstance(y, (pd.Series, np.ndarray)) else y[mask.values]
        
        if len(filtered_y) == 0:
            return None
        
        # Define luggage size ranges (percentiles)
        percentiles = [0, 25, 50, 75, 100]
        range_bounds = np.percentile(filtered_y, percentiles)
        
        # Calculate probabilities for each range
        probabilities = []
        for i in range(len(range_bounds) - 1):
            count = np.sum((filtered_y >= range_bounds[i]) & (filtered_y < range_bounds[i+1]))
            prob = count / len(filtered_y) * 100
            probabilities.append({
                'range': f"{range_bounds[i]:.2f} - {range_bounds[i+1]:.2f}",
                'probability': float(prob),
                'count': int(count)
            })
        
        return {
            'conditions': feature_ranges,
            'total_samples': int(len(filtered_y)),
            'probabilities': probabilities
        }
    
    def generate_insights(self, X, y, feature_names):
        """Generate comprehensive rule-based insights"""
        if self.tree is None:
            self.train(X, y, feature_names)
        
        insights = {
            'rules': [],
            'feature_combinations': [],
            'conditional_probabilities': []
        }
        
        # Extract rules
        rules = self.extract_rules(X, y)
        insights['rules'] = rules[:20]  # Top 20 rules
        
        # Find important feature combinations
        important_features = set()
        for rule in rules[:10]:
            for condition in rule['conditions']:
                feature = condition.split(' <= ')[0].split(' > ')[0]
                important_features.add(feature)
        
        # Generate conditional probabilities for top feature combinations
        if len(important_features) >= 2:
            features_list = list(important_features)[:3]  # Top 3 features
            for i, feat1 in enumerate(features_list):
                for feat2 in features_list[i+1:]:
                    if feat1 in feature_names and feat2 in feature_names:
                        # Get value ranges for these features
                        idx1 = feature_names.index(feat1)
                        idx2 = feature_names.index(feat2)
                        
                        vals1 = X[:, idx1] if isinstance(X, np.ndarray) else X[feat1].values
                        vals2 = X[:, idx2] if isinstance(X, np.ndarray) else X[feat2].values
                        
                        # Use quartiles as ranges
                        q1_min, q1_max = np.percentile(vals1, [25, 75])
                        q2_min, q2_max = np.percentile(vals2, [25, 75])
                        
                        prob_result = self.get_conditional_probabilities(
                            X, y, {feat1: (q1_min, q1_max), feat2: (q2_min, q2_max)}
                        )
                        
                        if prob_result:
                            insights['conditional_probabilities'].append(prob_result)
                            insights['feature_combinations'].append({
                                'features': [feat1, feat2],
                                'ranges': {
                                    feat1: (float(q1_min), float(q1_max)),
                                    feat2: (float(q2_min), float(q2_max))
                                }
                            })
        
        return insights



