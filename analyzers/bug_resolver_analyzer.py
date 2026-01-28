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
