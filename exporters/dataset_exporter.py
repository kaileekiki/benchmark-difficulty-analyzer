"""
Dataset exporter for generating research dataset files.
Exports various CSV files for bug difficulty analysis.
"""

import pandas as pd
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime


class DatasetExporter:
    """
    Exports various dataset formats for research analysis.
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        """
        Initialize the dataset exporter.
        
        Args:
            data_dir: Directory to save exported datasets
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    def export_all_datasets(self, leaderboard_data: List[Dict[str, Any]], 
                           bug_data: pd.DataFrame) -> Dict[str, str]:
        """
        Export all required datasets.
        
        Args:
            leaderboard_data: List of model performance dictionaries
            bug_data: DataFrame with bug resolution data
            
        Returns:
            Dictionary mapping dataset names to file paths
        """
        exported_files = {}
        
        self.logger.info("Exporting all datasets...")
        
        # Dataset 1: Model × Bug Matrix
        path1 = self.export_model_bug_matrix(leaderboard_data, bug_data)
        exported_files['model_bug_matrix'] = path1
        
        # Dataset 2: Bug × Tier Resolution Rates
        path2 = self.export_bug_tier_resolution(leaderboard_data, bug_data)
        exported_files['bug_tier_resolution'] = path2
        
        # Dataset 3: Difficulty Bracket Summary
        path3 = self.export_difficulty_brackets(bug_data)
        exported_files['difficulty_brackets'] = path3
        
        # Dataset 4: Model Performance Profile
        path4 = self.export_model_performance_profiles(leaderboard_data, bug_data)
        exported_files['model_performance_profiles'] = path4
        
        # Dataset 5: Bug Consensus Analysis
        path5 = self.export_bug_consensus(bug_data)
        exported_files['bug_consensus'] = path5
        
        self.logger.info(f"All datasets exported to {self.data_dir}/")
        
        return exported_files
    
    def export_model_bug_matrix(self, leaderboard_data: List[Dict[str, Any]], 
                                bug_data: pd.DataFrame) -> str:
        """
        Export Dataset 1: Model × Bug Matrix.
        
        Args:
            leaderboard_data: List of model performance dictionaries
            bug_data: DataFrame with bug resolution data
            
        Returns:
            Path to exported file
        """
        self.logger.info("Exporting Model × Bug Matrix...")
        
        # Get unique bug IDs and sort them
        bug_ids = sorted(bug_data['bug_id'].unique())
        
        # Sort models by score descending
        sorted_models = sorted(leaderboard_data, key=lambda x: x['score'], reverse=True)
        
        # Create matrix
        matrix_data = []
        
        for model in sorted_models:
            model_id = model['model_id']
            model_name = model['model_name']
            
            # Get bug results for this model
            model_bugs = bug_data[bug_data['model_id'] == model_id]
            
            # Create row with model name and bug results
            row = {'model_name': model_name}
            
            for bug_id in bug_ids:
                bug_row = model_bugs[model_bugs['bug_id'] == bug_id]
                if not bug_row.empty:
                    row[bug_id] = int(bug_row.iloc[0]['resolved'])
                else:
                    row[bug_id] = 0
            
            matrix_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(matrix_data)
        
        # Save to CSV
        output_path = os.path.join(self.data_dir, 'model_bug_matrix.csv')
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"  Model × Bug Matrix saved to {output_path}")
        self.logger.info(f"  Shape: {len(sorted_models)} models × {len(bug_ids)} bugs")
        
        return output_path
    
    def export_bug_tier_resolution(self, leaderboard_data: List[Dict[str, Any]], 
                                   bug_data: pd.DataFrame) -> str:
        """
        Export Dataset 2: Bug × Tier Resolution Rates.
        
        Args:
            leaderboard_data: List of model performance dictionaries
            bug_data: DataFrame with bug resolution data
            
        Returns:
            Path to exported file
        """
        self.logger.info("Exporting Bug × Tier Resolution Rates...")
        
        # Define tiers (in 5% increments)
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
                    model_tiers[model['model_id']] = f"{tier_min}-{tier_max}"
                    break
        
        # Get unique bug IDs
        bug_ids = sorted(bug_data['bug_id'].unique())
        
        # Calculate statistics for each bug
        results = []
        
        for bug_id in bug_ids:
            bug_rows = bug_data[bug_data['bug_id'] == bug_id]
            
            total_models = len(leaderboard_data)
            total_resolved = bug_rows['resolved'].sum()
            overall_rate = (total_resolved / total_models * 100) if total_models > 0 else 0
            
            row = {
                'bug_id': bug_id,
                'total_resolved': int(total_resolved),
                'total_models': total_models,
                'overall_rate': f"{overall_rate:.2f}"
            }
            
            # Calculate for each tier
            for tier_min, tier_max in tier_ranges:
                tier_label = f"{tier_min}-{tier_max}"
                
                # Get models in this tier
                tier_model_ids = [mid for mid, tier in model_tiers.items() if tier == tier_label]
                tier_total = len(tier_model_ids)
                
                # Count resolved in this tier
                tier_resolved = 0
                if tier_total > 0:
                    tier_bug_rows = bug_rows[bug_rows['model_id'].isin(tier_model_ids)]
                    tier_resolved = int(tier_bug_rows['resolved'].sum())
                
                # Calculate rate
                if tier_total > 0:
                    tier_rate = (tier_resolved / tier_total * 100)
                    tier_rate_str = f"{tier_rate:.2f}"
                else:
                    tier_rate_str = "N/A"
                
                row[f'tier_{tier_label}_resolved'] = tier_resolved
                row[f'tier_{tier_label}_total'] = tier_total
                row[f'tier_{tier_label}_rate'] = tier_rate_str
            
            results.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        output_path = os.path.join(self.data_dir, 'bug_tier_resolution.csv')
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"  Bug × Tier Resolution Rates saved to {output_path}")
        self.logger.info(f"  {len(bug_ids)} bugs × {len(tier_ranges)} tiers")
        
        return output_path
    
    def export_difficulty_brackets(self, bug_data: pd.DataFrame) -> str:
        """
        Export Dataset 3: Difficulty Bracket Summary.
        
        Args:
            bug_data: DataFrame with bug resolution data
            
        Returns:
            Path to exported file
        """
        self.logger.info("Exporting Difficulty Bracket Summary...")
        
        # Calculate resolution rate for each bug
        bug_stats = bug_data.groupby('bug_id').agg({
            'resolved': ['sum', 'count']
        }).reset_index()
        bug_stats.columns = ['bug_id', 'resolved_count', 'total_models']
        bug_stats['resolution_rate'] = (bug_stats['resolved_count'] / bug_stats['total_models'] * 100)
        
        # Define difficulty brackets
        brackets = [
            ('90-100%', 90, 100),
            ('80-90%', 80, 90),
            ('70-80%', 70, 80),
            ('60-70%', 60, 70),
            ('50-60%', 50, 60),
            ('40-50%', 40, 50),
            ('30-40%', 30, 40),
            ('20-30%', 20, 30),
            ('10-20%', 10, 20),
            ('0-10%', 0, 10)
        ]
        
        results = []
        
        for bracket_name, min_rate, max_rate in brackets:
            if max_rate == 100:
                bracket_bugs = bug_stats[(bug_stats['resolution_rate'] >= min_rate) & 
                                        (bug_stats['resolution_rate'] <= max_rate)]
            else:
                bracket_bugs = bug_stats[(bug_stats['resolution_rate'] >= min_rate) & 
                                        (bug_stats['resolution_rate'] < max_rate)]
            
            bug_count = len(bracket_bugs)
            
            if bug_count > 0:
                avg_rate = bracket_bugs['resolution_rate'].mean()
                
                # Calculate top/bottom model success rates (simplified)
                # Top 25% of models vs bottom 25%
                top_rate = bracket_bugs['resolution_rate'].quantile(0.75)
                bottom_rate = bracket_bugs['resolution_rate'].quantile(0.25)
            else:
                avg_rate = 0
                top_rate = 0
                bottom_rate = 0
            
            results.append({
                'difficulty_bracket': bracket_name,
                'bug_count': bug_count,
                'avg_resolution_rate': f"{avg_rate:.1f}%",
                'top_models_success_rate': f"{top_rate:.1f}%",
                'bottom_models_success_rate': f"{bottom_rate:.1f}%"
            })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        output_path = os.path.join(self.data_dir, 'difficulty_brackets.csv')
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"  Difficulty Bracket Summary saved to {output_path}")
        
        return output_path
    
    def export_model_performance_profiles(self, leaderboard_data: List[Dict[str, Any]], 
                                         bug_data: pd.DataFrame) -> str:
        """
        Export Dataset 4: Model Performance Profile.
        
        Args:
            leaderboard_data: List of model performance dictionaries
            bug_data: DataFrame with bug resolution data
            
        Returns:
            Path to exported file
        """
        self.logger.info("Exporting Model Performance Profiles...")
        
        # Calculate resolution rate for each bug
        bug_stats = bug_data.groupby('bug_id').agg({
            'resolved': ['sum', 'count']
        }).reset_index()
        bug_stats.columns = ['bug_id', 'resolved_count', 'total_models']
        bug_stats['resolution_rate'] = (bug_stats['resolved_count'] / bug_stats['total_models'] * 100)
        
        # Define difficulty categories
        def categorize_difficulty(rate):
            if rate >= 75:
                return 'easy'
            elif rate >= 50:
                return 'medium'
            elif rate >= 25:
                return 'hard'
            else:
                return 'very_hard'
        
        bug_stats['difficulty'] = bug_stats['resolution_rate'].apply(categorize_difficulty)
        
        # Create difficulty mapping
        bug_difficulty = dict(zip(bug_stats['bug_id'], bug_stats['difficulty']))
        
        results = []
        
        for model in sorted(leaderboard_data, key=lambda x: x['score'], reverse=True):
            model_id = model['model_id']
            model_name = model['model_name']
            
            # Get bug results for this model
            model_bugs = bug_data[bug_data['model_id'] == model_id]
            
            # Calculate rates by difficulty
            total_bugs = len(model_bugs)
            
            easy_bugs = model_bugs[model_bugs['bug_id'].map(bug_difficulty) == 'easy']
            medium_bugs = model_bugs[model_bugs['bug_id'].map(bug_difficulty) == 'medium']
            hard_bugs = model_bugs[model_bugs['bug_id'].map(bug_difficulty) == 'hard']
            very_hard_bugs = model_bugs[model_bugs['bug_id'].map(bug_difficulty) == 'very_hard']
            
            easy_rate = (easy_bugs['resolved'].sum() / len(easy_bugs) * 100) if len(easy_bugs) > 0 else 0
            medium_rate = (medium_bugs['resolved'].sum() / len(medium_bugs) * 100) if len(medium_bugs) > 0 else 0
            hard_rate = (hard_bugs['resolved'].sum() / len(hard_bugs) * 100) if len(hard_bugs) > 0 else 0
            very_hard_rate = (very_hard_bugs['resolved'].sum() / len(very_hard_bugs) * 100) if len(very_hard_bugs) > 0 else 0
            
            # Calculate unique vs shared solves
            resolved_bugs = set(model_bugs[model_bugs['resolved'] == 1]['bug_id'])
            
            unique_solves = 0
            shared_solves = 0
            
            for bug_id in resolved_bugs:
                # Count how many other models also solved this bug
                other_solves = bug_data[(bug_data['bug_id'] == bug_id) & 
                                       (bug_data['model_id'] != model_id) & 
                                       (bug_data['resolved'] == 1)]
                
                if len(other_solves) == 0:
                    unique_solves += 1
                else:
                    shared_solves += 1
            
            results.append({
                'model_name': model_name,
                'overall_rate': f"{model['score']:.2f}",
                'easy_bugs_rate': f"{easy_rate:.1f}%",
                'medium_bugs_rate': f"{medium_rate:.1f}%",
                'hard_bugs_rate': f"{hard_rate:.1f}%",
                'very_hard_bugs_rate': f"{very_hard_rate:.1f}%",
                'unique_solves': unique_solves,
                'shared_solves': shared_solves
            })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        output_path = os.path.join(self.data_dir, 'model_performance_profiles.csv')
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"  Model Performance Profiles saved to {output_path}")
        
        return output_path
    
    def export_bug_consensus(self, bug_data: pd.DataFrame) -> str:
        """
        Export Dataset 5: Bug Consensus Analysis.
        
        Args:
            bug_data: DataFrame with bug resolution data
            
        Returns:
            Path to exported file
        """
        self.logger.info("Exporting Bug Consensus Analysis...")
        
        # Calculate statistics for each bug
        bug_stats = bug_data.groupby('bug_id').agg({
            'resolved': ['sum', 'count']
        }).reset_index()
        bug_stats.columns = ['bug_id', 'models_passed', 'total_models']
        bug_stats['models_failed'] = bug_stats['total_models'] - bug_stats['models_passed']
        bug_stats['resolution_rate'] = (bug_stats['models_passed'] / bug_stats['total_models'] * 100)
        
        # Determine consensus type
        def determine_consensus(rate):
            if rate == 100:
                return 'universal_pass'
            elif rate == 0:
                return 'universal_fail'
            elif rate >= 80:
                return 'high_agreement_pass'
            elif rate <= 20:
                return 'high_agreement_fail'
            elif 40 <= rate <= 60:
                return 'controversial'
            elif rate > 60:
                return 'moderate_agreement_pass'
            else:
                return 'moderate_agreement_fail'
        
        def determine_difficulty(rate):
            if rate >= 75:
                return 'Easy'
            elif rate >= 50:
                return 'Medium'
            elif rate >= 25:
                return 'Hard'
            else:
                return 'Very Hard'
        
        bug_stats['consensus_type'] = bug_stats['resolution_rate'].apply(determine_consensus)
        bug_stats['difficulty_label'] = bug_stats['resolution_rate'].apply(determine_difficulty)
        
        # Format resolution_rate
        bug_stats['resolution_rate'] = bug_stats['resolution_rate'].apply(lambda x: f"{x:.1f}%")
        
        # Select and reorder columns
        df = bug_stats[['bug_id', 'consensus_type', 'models_passed', 'models_failed', 
                       'resolution_rate', 'difficulty_label']]
        
        # Convert to int
        df['models_passed'] = df['models_passed'].astype(int)
        df['models_failed'] = df['models_failed'].astype(int)
        
        # Save to CSV
        output_path = os.path.join(self.data_dir, 'bug_consensus.csv')
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"  Bug Consensus Analysis saved to {output_path}")
        
        # Log statistics
        consensus_counts = df['consensus_type'].value_counts()
        self.logger.info(f"  Consensus breakdown:")
        for consensus, count in consensus_counts.items():
            self.logger.info(f"    {consensus}: {count} bugs")
        
        return output_path
