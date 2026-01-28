"""
Model comparison and agreement analysis.
Analyzes similarities and differences between models.
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class ModelComparison:
    """
    Compares models to find similarities, differences, and clusters.
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        """
        Initialize the model comparison analyzer.
        
        Args:
            data_dir: Directory for processed data
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        os.makedirs(data_dir, exist_ok=True)
    
    def calculate_pairwise_agreement(self, bug_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pairwise agreement between all models.
        
        Args:
            bug_data: DataFrame with bug resolution data
            
        Returns:
            DataFrame with pairwise agreement scores
        """
        models = bug_data['model_id'].unique()
        agreements = []
        
        for i, model1 in enumerate(models):
            for model2 in models[i:]:  # Include self-comparison
                agreement = self._calculate_agreement(bug_data, model1, model2)
                
                agreements.append({
                    'model1': model1,
                    'model2': model2,
                    'agreement_rate': agreement['agreement_rate'],
                    'both_solved': agreement['both_solved'],
                    'both_failed': agreement['both_failed'],
                    'disagreement': agreement['disagreement'],
                    'total_bugs': agreement['total_bugs']
                })
        
        agreement_df = pd.DataFrame(agreements)
        self.logger.info(f"Calculated {len(agreement_df)} pairwise agreements")
        
        return agreement_df
    
    def _calculate_agreement(self, bug_data: pd.DataFrame, 
                            model1: str, model2: str) -> Dict[str, float]:
        """
        Calculate agreement between two models.
        
        Args:
            bug_data: DataFrame with bug resolution data
            model1: First model ID
            model2: Second model ID
            
        Returns:
            Dictionary with agreement statistics
        """
        # Get results for both models
        m1_data = bug_data[bug_data['model_id'] == model1].set_index('bug_id')['resolved']
        m2_data = bug_data[bug_data['model_id'] == model2].set_index('bug_id')['resolved']
        
        # Find common bugs
        common_bugs = m1_data.index.intersection(m2_data.index)
        
        if len(common_bugs) == 0:
            return {
                'agreement_rate': 0.0,
                'both_solved': 0,
                'both_failed': 0,
                'disagreement': 0,
                'total_bugs': 0
            }
        
        m1_results = m1_data.loc[common_bugs]
        m2_results = m2_data.loc[common_bugs]
        
        # Calculate agreement
        both_solved = ((m1_results == 1) & (m2_results == 1)).sum()
        both_failed = ((m1_results == 0) & (m2_results == 0)).sum()
        disagreement = ((m1_results != m2_results)).sum()
        
        agreement_rate = (both_solved + both_failed) / len(common_bugs)
        
        return {
            'agreement_rate': agreement_rate,
            'both_solved': int(both_solved),
            'both_failed': int(both_failed),
            'disagreement': int(disagreement),
            'total_bugs': len(common_bugs)
        }
    
    def find_model_clusters(self, agreement_df: pd.DataFrame, 
                           threshold: float = 0.8) -> List[List[str]]:
        """
        Find clusters of models with high agreement.
        
        Args:
            agreement_df: DataFrame with pairwise agreement scores
            threshold: Minimum agreement rate to consider models similar
            
        Returns:
            List of model clusters (lists of model IDs)
        """
        # Build similarity graph
        high_agreement = agreement_df[
            (agreement_df['agreement_rate'] >= threshold) & 
            (agreement_df['model1'] != agreement_df['model2'])
        ]
        
        # Simple clustering: group models with high mutual agreement
        clusters = []
        processed = set()
        
        for model in agreement_df['model1'].unique():
            if model in processed:
                continue
            
            # Find models with high agreement to this model
            similar = high_agreement[
                (high_agreement['model1'] == model) | 
                (high_agreement['model2'] == model)
            ]
            
            cluster_models = set([model])
            cluster_models.update(similar['model1'].unique())
            cluster_models.update(similar['model2'].unique())
            
            clusters.append(list(cluster_models))
            processed.update(cluster_models)
        
        self.logger.info(f"Found {len(clusters)} model clusters with threshold {threshold}")
        return clusters
    
    def generate_performance_matrix(self, bug_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate bugs × models performance matrix.
        
        Args:
            bug_data: DataFrame with bug resolution data
            
        Returns:
            DataFrame with bugs as rows, models as columns, resolution status as values
        """
        # Pivot to create matrix
        matrix = bug_data.pivot(
            index='bug_id',
            columns='model_id',
            values='resolved'
        )
        
        # Fill NaN with 0 (not attempted or failed)
        matrix = matrix.fillna(0)
        
        self.logger.info(f"Generated performance matrix: {matrix.shape[0]} bugs × {matrix.shape[1]} models")
        return matrix
    
    def analyze_model_strengths(self, bug_data: pd.DataFrame,
                                bug_categories: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Analyze model strengths by bug category/type.
        
        Args:
            bug_data: DataFrame with bug resolution data
            bug_categories: Optional DataFrame mapping bug_id to category
            
        Returns:
            DataFrame with model performance by category
        """
        if bug_categories is None:
            # If no categories provided, use difficulty as category
            from .bug_resolver_analyzer import BugResolverAnalyzer
            
            # Create simple categories based on bug_id patterns
            self.logger.warning("No bug categories provided, skipping category analysis")
            return pd.DataFrame()
        
        # Merge bug data with categories
        categorized = bug_data.merge(bug_categories, on='bug_id', how='left')
        
        # Analyze by model and category
        strengths = categorized.groupby(['model_id', 'category']).agg({
            'resolved': ['sum', 'count', 'mean']
        }).reset_index()
        
        strengths.columns = ['model_id', 'category', 'solved', 'attempted', 'success_rate']
        
        self.logger.info(f"Analyzed strengths for {strengths['model_id'].nunique()} models")
        return strengths
    
    def identify_unique_solvers(self, bug_data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify bugs that only specific models can solve.
        
        Args:
            bug_data: DataFrame with bug resolution data
            
        Returns:
            DataFrame with unique solver information
        """
        # For each bug, find which models solved it
        bug_solvers = bug_data[bug_data['resolved'] == 1].groupby('bug_id').agg({
            'model_id': lambda x: list(x),
            'resolved': 'count'
        }).reset_index()
        
        bug_solvers.columns = ['bug_id', 'solver_models', 'num_solvers']
        
        # Identify unique solvers (only 1 model solved it)
        unique_solvers = bug_solvers[bug_solvers['num_solvers'] == 1].copy()
        unique_solvers['unique_solver'] = unique_solvers['solver_models'].apply(lambda x: x[0])
        
        self.logger.info(f"Found {len(unique_solvers)} bugs with unique solvers")
        
        return unique_solvers[['bug_id', 'unique_solver']]
    
    def compare_model_pairs(self, bug_data: pd.DataFrame,
                           model_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Detailed comparison of specific model pairs.
        
        Args:
            bug_data: DataFrame with bug resolution data
            model_pairs: List of (model1, model2) tuples to compare
            
        Returns:
            DataFrame with detailed comparison
        """
        comparisons = []
        
        for model1, model2 in model_pairs:
            # Get agreement
            agreement = self._calculate_agreement(bug_data, model1, model2)
            
            # Get unique bugs each model solves
            m1_solved = set(bug_data[
                (bug_data['model_id'] == model1) & 
                (bug_data['resolved'] == 1)
            ]['bug_id'])
            
            m2_solved = set(bug_data[
                (bug_data['model_id'] == model2) & 
                (bug_data['resolved'] == 1)
            ]['bug_id'])
            
            only_m1 = len(m1_solved - m2_solved)
            only_m2 = len(m2_solved - m1_solved)
            
            comparisons.append({
                'model1': model1,
                'model2': model2,
                'agreement_rate': agreement['agreement_rate'],
                'model1_unique_solves': only_m1,
                'model2_unique_solves': only_m2,
                'both_solved': agreement['both_solved'],
                'both_failed': agreement['both_failed']
            })
        
        comparison_df = pd.DataFrame(comparisons)
        self.logger.info(f"Compared {len(comparison_df)} model pairs")
        
        return comparison_df
    
    def save_analysis(self, data: pd.DataFrame, filename: str):
        """
        Save analysis results.
        
        Args:
            data: DataFrame to save
            filename: Output filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        csv_path = os.path.join(self.data_dir, f'{filename}_{timestamp}.csv')
        data.to_csv(csv_path, index=False)
        
        json_path = os.path.join(self.data_dir, f'{filename}_{timestamp}.json')
        data.to_json(json_path, orient='records', indent=2)
        
        self.logger.info(f"Saved analysis to {csv_path} and {json_path}")
