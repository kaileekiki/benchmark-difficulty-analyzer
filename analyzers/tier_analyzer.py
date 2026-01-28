"""
Performance tier analysis.
Groups models by performance and analyzes bug-solving patterns within tiers.
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any


class TierAnalyzer:
    """
    Analyzes model performance tiers and bug-solving patterns.
    Groups models by performance ranges and calculates similarity metrics.
    """
    
    def __init__(self, tier_ranges: List[Tuple[float, float]], data_dir: str = "data/processed"):
        """
        Initialize the tier analyzer.
        
        Args:
            tier_ranges: List of (min, max) tuples defining performance tiers
            data_dir: Directory for processed data
        """
        self.tier_ranges = tier_ranges
        self.data_dir = data_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        os.makedirs(data_dir, exist_ok=True)
    
    def assign_models_to_tiers(self, leaderboard_data: List[Dict]) -> pd.DataFrame:
        """
        Assign models to performance tiers.
        
        Args:
            leaderboard_data: List of model performance dictionaries
            
        Returns:
            DataFrame with model tier assignments
        """
        models_df = pd.DataFrame(leaderboard_data)
        
        # Assign tier based on score
        def get_tier(score):
            for i, (min_score, max_score) in enumerate(self.tier_ranges):
                if min_score <= score < max_score:
                    return f"Tier_{i+1}_{min_score}-{max_score}%"
            return "Tier_Other"
        
        models_df['tier'] = models_df['score'].apply(get_tier)
        
        self.logger.info(f"Assigned {len(models_df)} models to tiers")
        self.logger.info(f"Tier distribution:\n{models_df['tier'].value_counts()}")
        
        return models_df
    
    def analyze_tier_overlap(self, bug_data: pd.DataFrame, 
                            tier_assignments: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze bug-solving overlap within performance tiers.
        
        Args:
            bug_data: DataFrame with bug resolution data
            tier_assignments: DataFrame with model tier assignments
            
        Returns:
            DataFrame with tier overlap statistics
        """
        # Merge bug data with tier assignments
        bug_tier_data = bug_data.merge(
            tier_assignments[['model_id', 'tier']], 
            on='model_id'
        )
        
        # Analyze by tier
        tier_stats = []
        
        for tier in tier_assignments['tier'].unique():
            tier_bugs = bug_tier_data[bug_tier_data['tier'] == tier]
            
            # Count models in tier
            models_in_tier = tier_assignments[tier_assignments['tier'] == tier]['model_id'].nunique()
            
            # Calculate statistics
            total_attempts = len(tier_bugs)
            resolved = tier_bugs['resolved'].sum()
            resolution_rate = resolved / total_attempts if total_attempts > 0 else 0
            
            # Get unique bugs attempted
            unique_bugs = tier_bugs['bug_id'].nunique()
            
            tier_stats.append({
                'tier': tier,
                'num_models': models_in_tier,
                'total_attempts': total_attempts,
                'resolved': resolved,
                'resolution_rate': resolution_rate,
                'unique_bugs': unique_bugs
            })
        
        tier_stats_df = pd.DataFrame(tier_stats)
        self.logger.info(f"Analyzed {len(tier_stats_df)} tiers")
        
        return tier_stats_df
    
    def calculate_jaccard_similarity(self, bug_data: pd.DataFrame,
                                     tier_assignments: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pairwise Jaccard similarity between models in same tier.
        
        Args:
            bug_data: DataFrame with bug resolution data
            tier_assignments: DataFrame with model tier assignments
            
        Returns:
            DataFrame with pairwise similarity scores
        """
        similarities = []
        
        # Merge bug data with tiers
        bug_tier_data = bug_data.merge(
            tier_assignments[['model_id', 'tier']], 
            on='model_id'
        )
        
        # For each tier
        for tier in tier_assignments['tier'].unique():
            tier_models = tier_assignments[tier_assignments['tier'] == tier]['model_id'].unique()
            
            if len(tier_models) < 2:
                continue
            
            # Calculate pairwise similarity
            for i, model1 in enumerate(tier_models):
                for model2 in tier_models[i+1:]:
                    similarity = self._jaccard_similarity(
                        bug_tier_data, model1, model2
                    )
                    
                    similarities.append({
                        'tier': tier,
                        'model1': model1,
                        'model2': model2,
                        'jaccard_similarity': similarity
                    })
        
        similarity_df = pd.DataFrame(similarities)
        self.logger.info(f"Calculated {len(similarity_df)} pairwise similarities")
        
        return similarity_df
    
    def _jaccard_similarity(self, bug_data: pd.DataFrame, 
                           model1: str, model2: str) -> float:
        """
        Calculate Jaccard similarity between two models' solved bugs.
        
        Args:
            bug_data: DataFrame with bug resolution data
            model1: First model ID
            model2: Second model ID
            
        Returns:
            Jaccard similarity score (0-1)
        """
        # Get bugs solved by each model
        m1_bugs = set(bug_data[
            (bug_data['model_id'] == model1) & 
            (bug_data['resolved'] == 1)
        ]['bug_id'])
        
        m2_bugs = set(bug_data[
            (bug_data['model_id'] == model2) & 
            (bug_data['resolved'] == 1)
        ]['bug_id'])
        
        # Calculate Jaccard similarity
        if len(m1_bugs) == 0 and len(m2_bugs) == 0:
            return 1.0
        
        intersection = len(m1_bugs & m2_bugs)
        union = len(m1_bugs | m2_bugs)
        
        return intersection / union if union > 0 else 0.0
    
    def identify_common_bugs_by_tier(self, bug_data: pd.DataFrame,
                                     tier_assignments: pd.DataFrame,
                                     min_models: int = 2) -> Dict[str, List[str]]:
        """
        Identify bugs commonly solved or failed within each tier.
        
        Args:
            bug_data: DataFrame with bug resolution data
            tier_assignments: DataFrame with model tier assignments
            min_models: Minimum number of models that must agree
            
        Returns:
            Dictionary mapping tiers to lists of common bugs
        """
        # Merge data
        bug_tier_data = bug_data.merge(
            tier_assignments[['model_id', 'tier']], 
            on='model_id'
        )
        
        common_bugs = {}
        
        for tier in tier_assignments['tier'].unique():
            tier_bugs = bug_tier_data[bug_tier_data['tier'] == tier]
            
            # Group by bug and count how many models solved it
            bug_counts = tier_bugs.groupby('bug_id').agg({
                'resolved': ['sum', 'count']
            }).reset_index()
            
            bug_counts.columns = ['bug_id', 'solved_count', 'total_count']
            
            # Find bugs solved by most models in tier
            commonly_solved = bug_counts[
                bug_counts['solved_count'] >= min_models
            ]['bug_id'].tolist()
            
            # Find bugs failed by most models in tier
            commonly_failed = bug_counts[
                (bug_counts['total_count'] - bug_counts['solved_count']) >= min_models
            ]['bug_id'].tolist()
            
            common_bugs[tier] = {
                'commonly_solved': commonly_solved,
                'commonly_failed': commonly_failed
            }
        
        self.logger.info(f"Identified common bugs for {len(common_bugs)} tiers")
        
        return common_bugs
    
    def save_analysis(self, data: Any, filename: str):
        """
        Save analysis results.
        
        Args:
            data: Data to save (DataFrame or dict)
            filename: Output filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if isinstance(data, pd.DataFrame):
            csv_path = os.path.join(self.data_dir, f'{filename}_{timestamp}.csv')
            data.to_csv(csv_path, index=False)
            self.logger.info(f"Saved to {csv_path}")
        
        # Always save JSON
        json_path = os.path.join(self.data_dir, f'{filename}_{timestamp}.json')
        
        if isinstance(data, pd.DataFrame):
            data.to_json(json_path, orient='records', indent=2)
        else:
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved to {json_path}")
