"""
Bug-level resolution analysis.
Analyzes which bugs are solved by which models and categorizes difficulty.
"""

import pandas as pd
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from difficulty.metrics import DifficultyMetrics
from data.loaders import SWEBenchVerifiedLoader


class BugResolverAnalyzer:
    """
    Analyzes bug resolution rates across models.
    Identifies hardest/easiest bugs and calculates difficulty metrics.
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        """
        Initialize the analyzer.
        
        Args:
            data_dir: Directory containing processed data
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        self.bug_data = None
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def load_bug_data(self, bug_data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load bug resolution data.
        
        Args:
            bug_data_path: Path to bug data CSV file
            
        Returns:
            DataFrame with bug resolution data
        """
        if bug_data_path is None:
            bug_data_path = os.path.join(self.data_dir, 'bug_data.csv')
        
        try:
            self.bug_data = pd.read_csv(bug_data_path)
            self.logger.info(f"Loaded bug data from {bug_data_path}")
            return self.bug_data
        except FileNotFoundError:
            self.logger.error(f"Bug data file not found: {bug_data_path}")
            raise
    
    def create_bug_data_from_crawled(self, leaderboard_data: List[Dict], 
                                      bug_results: Dict[str, Dict[str, bool]]) -> pd.DataFrame:
        """
        Create bug data DataFrame from crawled leaderboard and bug results.
        
        Args:
            leaderboard_data: List of model performance dictionaries
            bug_results: Dictionary mapping model_id to bug results
            
        Returns:
            DataFrame with bug resolution data
        """
        rows = []
        
        for model_data in leaderboard_data:
            model_id = model_data['model_id']
            model_name = model_data['model_name']
            
            # Get bug results for this model
            model_bugs = bug_results.get(model_id, {})
            
            for bug_id, resolved in model_bugs.items():
                rows.append({
                    'model_id': model_id,
                    'model_name': model_name,
                    'bug_id': bug_id,
                    'resolved': 1 if resolved else 0
                })
        
        self.bug_data = pd.DataFrame(rows)
        self.logger.info(f"Created bug data with {len(self.bug_data)} records")
        
        # Save the bug data (no timestamp - this is the current dataset for analysis)
        # Other save_analysis() calls use timestamps for historical tracking,
        # but bug_data.csv is the working dataset that should be overwritten
        bug_data_path = os.path.join(self.data_dir, 'bug_data.csv')
        self.bug_data.to_csv(bug_data_path, index=False)
        
        return self.bug_data

    def analyze_bug_resolution_rates(self) -> pd.DataFrame:
        """
        Calculate resolution rate for each bug across all models.
        
        Returns:
            DataFrame with bug resolution statistics
        """
        if self.bug_data is None:
            raise ValueError("Bug data not loaded. Call load_bug_data() first.")
        
        analysis = self.bug_data.groupby('bug_id').agg(
            total_models=('model_id', 'count'),
            resolved_count=('resolved', 'sum')
        ).reset_index()
        
        analysis['resolution_rate'] = analysis['resolved_count'] / analysis['total_models']
        analysis['difficulty_label'] = analysis['resolution_rate'].apply(self._categorize_difficulty)
        
        # Sort by resolution rate (hardest first)
        analysis = analysis.sort_values('resolution_rate')
        
        self.logger.info(f"Analyzed {len(analysis)} bugs")
        return analysis[['bug_id', 'total_models', 'resolved_count', 'resolution_rate', 'difficulty_label']]

    def _categorize_difficulty(self, resolution_rate: float) -> str:
        """
        Categorize bug difficulty based on resolution rate.
        
        Args:
            resolution_rate: Percentage of models that solved the bug (0-1)
            
        Returns:
            Difficulty category string
        """
        if resolution_rate >= 0.75:
            return 'Easy'
        elif resolution_rate >= 0.5:
            return 'Medium'
        elif resolution_rate >= 0.25:
            return 'Hard'
        else:
            return 'Very Hard'

    def identify_hardest_bugs(self, threshold: float = 0.25) -> pd.DataFrame:
        """
        Identify the hardest bugs (low resolution rate).
        
        Args:
            threshold: Resolution rate threshold (bugs below this are "hardest")
            
        Returns:
            DataFrame with hardest bugs
        """
        analysis = self.analyze_bug_resolution_rates()
        hardest = analysis[analysis['resolution_rate'] < threshold]
        self.logger.info(f"Found {len(hardest)} bugs with resolution rate < {threshold}")
        return hardest

    def identify_easiest_bugs(self) -> pd.DataFrame:
        """
        Identify the easiest bugs (high resolution rate).
        
        Returns:
            DataFrame with easiest bugs
        """
        analysis = self.analyze_bug_resolution_rates()
        easiest = analysis[analysis['difficulty_label'] == 'Easy']
        self.logger.info(f"Found {len(easiest)} easy bugs")
        return easiest

    def analyze_consensus_bugs(self) -> pd.DataFrame:
        """
        Analyze bugs where all models agree (all pass or all fail).
        
        Returns:
            DataFrame with consensus statistics
        """
        if self.bug_data is None:
            raise ValueError("Bug data not loaded. Call load_bug_data() first.")
        
        consensus = self.bug_data.groupby('bug_id').agg(
            total_models=('resolved', 'count'),
            resolved_count=('resolved', 'sum'),
            consensus_rate=('resolved', 'mean')
        ).reset_index()
        
        # Identify perfect consensus (all pass or all fail)
        consensus['all_pass'] = consensus['consensus_rate'] == 1.0
        consensus['all_fail'] = consensus['consensus_rate'] == 0.0
        consensus['consensus'] = consensus['all_pass'] | consensus['all_fail']
        
        self.logger.info(
            f"Found {consensus['all_pass'].sum()} bugs all models solved, "
            f"{consensus['all_fail'].sum()} bugs no model solved"
        )
        
        return consensus

    def get_difficulty_distribution(self) -> Dict[str, int]:
        """
        Get count of bugs in each difficulty category.
        
        Returns:
            Dictionary mapping difficulty labels to counts
        """
        analysis = self.analyze_bug_resolution_rates()
        distribution = analysis['difficulty_label'].value_counts().to_dict()
        
        self.logger.info(f"Difficulty distribution: {distribution}")
        return distribution
    
    def calculate_multiple_difficulties(self, leaderboard_data: List[Dict], 
                                       difficulty_config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate multiple difficulty metrics for each bug.
        
        Args:
            leaderboard_data: List of model performance dictionaries for tier calculation
            difficulty_config: Configuration dictionary with weights
            
        Returns:
            DataFrame with bug_id and multiple difficulty columns
        """
        if self.bug_data is None:
            raise ValueError("Bug data not loaded. Call create_bug_data_from_crawled() first.")
        
        self.logger.info("Calculating multiple difficulty metrics...")
        
        # Get basic resolution rates
        resolution_data = self.analyze_bug_resolution_rates()
        
        # Load time-based difficulty from HuggingFace
        try:
            loader = SWEBenchVerifiedLoader()
            time_difficulties = loader.get_all_difficulties()
            self.logger.info(f"Loaded time difficulties for {len(time_difficulties)} bugs")
        except Exception as e:
            self.logger.warning(f"Could not load time difficulties: {e}")
            time_difficulties = {}
        
        # Calculate tier-based difficulties
        tier_difficulties = self._calculate_tier_difficulties(leaderboard_data)
        
        # Get weights from config or use defaults
        if difficulty_config:
            weights = difficulty_config.get('weights', {'success_rate': 0.4, 'time': 0.6})
        else:
            weights = {'success_rate': 0.4, 'time': 0.6}
        
        # Calculate all difficulty metrics
        results = []
        for _, row in resolution_data.iterrows():
            bug_id = row['bug_id']
            success_rate = row['resolution_rate']
            
            # 1. Success rate-based difficulty
            diff_success = DifficultyMetrics.from_model_success_rate(success_rate)
            
            # 2. Time-based difficulty
            time_str = time_difficulties.get(bug_id)
            diff_time = DifficultyMetrics.from_time_difficulty(time_str) if time_str else 0.5
            
            # 3. Tier-based difficulty
            tier_results = tier_difficulties.get(bug_id, {})
            diff_tier = DifficultyMetrics.from_model_tiers(tier_results) if tier_results else 0.5
            
            # 4. Combined difficulty
            diff_combined = DifficultyMetrics.combined(
                success_rate, 
                diff_time, 
                diff_tier,
                weights
            )
            
            results.append({
                'bug_id': bug_id,
                'resolution_rate': success_rate,
                'difficulty_success_rate': diff_success,
                'difficulty_time': diff_time,
                'difficulty_tier': diff_tier,
                'difficulty_combined': diff_combined,
                'time_difficulty_str': time_str if time_str else 'N/A'
            })
        
        df = pd.DataFrame(results)
        self.logger.info(f"Calculated {len(df)} bug difficulty metrics")
        
        return df
    
    def _calculate_tier_difficulties(self, leaderboard_data: List[Dict]) -> Dict[str, Dict[str, float]]:
        """
        Calculate tier-based resolution rates for each bug.
        
        Args:
            leaderboard_data: List of model performance dictionaries
            
        Returns:
            Dictionary mapping bug_id to tier results
        """
        if self.bug_data is None:
            return {}
        
        # Define tier ranges
        tier_ranges = [
            (95, 100), (90, 95), (85, 90), (80, 85), (75, 80),
            (70, 75), (65, 70), (60, 65), (55, 60), (50, 55),
            (45, 50), (40, 45), (35, 40), (30, 35), (25, 30),
            (20, 25), (15, 20), (10, 15), (5, 10), (0, 5)
        ]
        
        # Assign models to tiers
        model_tiers = {}
        for model in leaderboard_data:
            score = model['score']
            for tier_min, tier_max in tier_ranges:
                if tier_min <= score < tier_max or (tier_max == 100 and score == 100):
                    tier_label = f"{tier_min}-{tier_max}"
                    model_tiers[model['model_id']] = tier_label
                    break
        
        # Calculate resolution rates per tier for each bug
        bug_tier_results = {}
        bug_ids = self.bug_data['bug_id'].unique()
        
        for bug_id in bug_ids:
            bug_rows = self.bug_data[self.bug_data['bug_id'] == bug_id]
            tier_results = {}
            
            for tier_min, tier_max in tier_ranges:
                tier_label = f"{tier_min}-{tier_max}"
                
                # Get models in this tier
                tier_model_ids = [mid for mid, tier in model_tiers.items() if tier == tier_label]
                
                if tier_model_ids:
                    tier_bug_rows = bug_rows[bug_rows['model_id'].isin(tier_model_ids)]
                    if len(tier_bug_rows) > 0:
                        tier_rate = tier_bug_rows['resolved'].sum() / len(tier_bug_rows)
                        tier_results[tier_label] = tier_rate
            
            bug_tier_results[bug_id] = tier_results
        
        return bug_tier_results

    def save_analysis(self, analysis: pd.DataFrame, filename_prefix: str):
        """
        Save analysis results to CSV and JSON.
        
        Args:
            analysis: DataFrame to save
            filename_prefix: Prefix for output filenames
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save CSV
        csv_path = os.path.join(self.data_dir, f'{filename_prefix}_{timestamp}.csv')
        analysis.to_csv(csv_path, index=False)
        self.logger.info(f"Saved CSV to {csv_path}")
        
        # Save JSON
        json_path = os.path.join(self.data_dir, f'{filename_prefix}_{timestamp}.json')
        analysis.to_json(json_path, orient='records', indent=2)
        self.logger.info(f"Saved JSON to {json_path}")
