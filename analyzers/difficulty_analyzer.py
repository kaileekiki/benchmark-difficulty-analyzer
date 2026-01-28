"""
Difficulty metric correlation analysis.
Correlates bug metrics (AST-GED, DFG-GED, PDG-GED, LOC) with resolution difficulty.
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from scipy.stats import pearsonr, spearmanr


class DifficultyAnalyzer:
    """
    Analyzes correlation between bug metrics and difficulty.
    Answers research questions about which metrics best predict difficulty.
    """
    
    def __init__(self, data_dir: str = "data/processed"):
        """
        Initialize the difficulty analyzer.
        
        Args:
            data_dir: Directory for processed data
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics_data = None
        self.resolution_data = None
        
        os.makedirs(data_dir, exist_ok=True)
    
    def load_metrics(self, metrics_path: str) -> pd.DataFrame:
        """
        Load bug metrics from CSV file.
        
        Expected columns: bug_id, ast_ged, dfg_ged, pdg_ged, loc, cyclomatic_complexity, etc.
        
        Args:
            metrics_path: Path to metrics CSV file
            
        Returns:
            DataFrame with bug metrics
        """
        try:
            self.metrics_data = pd.read_csv(metrics_path)
            self.logger.info(f"Loaded metrics for {len(self.metrics_data)} bugs from {metrics_path}")
            return self.metrics_data
        except FileNotFoundError:
            self.logger.error(f"Metrics file not found: {metrics_path}")
            raise
    
    def merge_with_resolution_data(self, resolution_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge bug metrics with resolution rates.
        
        Args:
            resolution_data: DataFrame with bug_id and resolution_rate
            
        Returns:
            Merged DataFrame
        """
        self.resolution_data = resolution_data
        
        # Merge on bug_id
        merged = self.metrics_data.merge(
            resolution_data[['bug_id', 'resolution_rate', 'difficulty_label']], 
            on='bug_id', 
            how='inner'
        )
        
        self.logger.info(f"Merged data contains {len(merged)} bugs")
        return merged
    
    def calculate_correlations(self, merged_data: pd.DataFrame,
                               metric_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate correlations between metrics and resolution difficulty.
        
        Args:
            merged_data: DataFrame with metrics and resolution_rate
            metric_columns: List of metric column names (auto-detect if None)
            
        Returns:
            DataFrame with correlation results
        """
        if metric_columns is None:
            # Auto-detect metric columns (numeric columns except bug_id and resolution_rate)
            metric_columns = [
                col for col in merged_data.select_dtypes(include=[np.number]).columns
                if col not in ['resolution_rate'] and 'id' not in col.lower()
            ]
        
        self.logger.info(f"Calculating correlations for {len(metric_columns)} metrics")
        
        correlations = []
        
        for metric in metric_columns:
            # Skip if metric has missing values
            valid_data = merged_data[[metric, 'resolution_rate']].dropna()
            
            if len(valid_data) < 2:
                self.logger.warning(f"Insufficient data for metric: {metric}")
                continue
            
            # Calculate Pearson correlation
            try:
                pearson_corr, pearson_pval = pearsonr(
                    valid_data[metric], 
                    valid_data['resolution_rate']
                )
            except Exception as e:
                self.logger.debug(f"Error calculating Pearson correlation for {metric}: {e}")
                pearson_corr, pearson_pval = np.nan, np.nan
            
            # Calculate Spearman correlation (rank-based, more robust)
            try:
                spearman_corr, spearman_pval = spearmanr(
                    valid_data[metric], 
                    valid_data['resolution_rate']
                )
            except Exception as e:
                self.logger.debug(f"Error calculating Spearman correlation for {metric}: {e}")
                spearman_corr, spearman_pval = np.nan, np.nan
            
            correlations.append({
                'metric': metric,
                'pearson_correlation': pearson_corr,
                'pearson_pvalue': pearson_pval,
                'spearman_correlation': spearman_corr,
                'spearman_pvalue': spearman_pval,
                'sample_size': len(valid_data)
            })
        
        corr_df = pd.DataFrame(correlations)
        
        # Sort by absolute correlation strength
        corr_df['abs_spearman'] = corr_df['spearman_correlation'].abs()
        corr_df = corr_df.sort_values('abs_spearman', ascending=False)
        
        self.logger.info("Top 3 predictive metrics:")
        for _, row in corr_df.head(3).iterrows():
            self.logger.info(f"  {row['metric']}: ρ={row['spearman_correlation']:.3f} (p={row['spearman_pvalue']:.4f})")
        
        return corr_df
    
    def analyze_by_difficulty_category(self, merged_data: pd.DataFrame,
                                       metric_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze metric distributions by difficulty category.
        
        Args:
            merged_data: DataFrame with metrics and difficulty_label
            metric_columns: List of metric column names
            
        Returns:
            DataFrame with statistics by category
        """
        if metric_columns is None:
            metric_columns = [
                col for col in merged_data.select_dtypes(include=[np.number]).columns
                if col not in ['resolution_rate'] and 'id' not in col.lower()
            ]
        
        category_stats = []
        
        for category in ['Easy', 'Medium', 'Hard', 'Very Hard']:
            category_data = merged_data[merged_data['difficulty_label'] == category]
            
            if len(category_data) == 0:
                continue
            
            stats = {'difficulty_category': category, 'count': len(category_data)}
            
            for metric in metric_columns:
                if metric in category_data.columns:
                    stats[f'{metric}_mean'] = category_data[metric].mean()
                    stats[f'{metric}_median'] = category_data[metric].median()
                    stats[f'{metric}_std'] = category_data[metric].std()
            
            category_stats.append(stats)
        
        stats_df = pd.DataFrame(category_stats)
        self.logger.info(f"Analyzed metrics across {len(stats_df)} difficulty categories")
        
        return stats_df
    
    def identify_top_predictors(self, correlation_df: pd.DataFrame, 
                                top_n: int = 5) -> List[Dict[str, float]]:
        """
        Identify top N difficulty predictor metrics.
        
        Args:
            correlation_df: DataFrame with correlation results
            top_n: Number of top predictors to return
            
        Returns:
            List of top predictor dictionaries
        """
        # Filter for significant correlations (p < 0.05)
        significant = correlation_df[correlation_df['spearman_pvalue'] < 0.05]
        
        # Get top N by absolute correlation
        top_predictors = significant.nlargest(top_n, 'abs_spearman')
        
        predictors = []
        for _, row in top_predictors.iterrows():
            predictors.append({
                'metric': row['metric'],
                'correlation': row['spearman_correlation'],
                'pvalue': row['spearman_pvalue'],
                'strength': 'strong' if abs(row['spearman_correlation']) > 0.7 
                           else 'moderate' if abs(row['spearman_correlation']) > 0.4 
                           else 'weak'
            })
        
        self.logger.info(f"Identified {len(predictors)} top predictors")
        return predictors
    
    def answer_rq1(self, correlation_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Answer RQ1: Which graph representation best predicts LLM repair difficulty?
        
        Args:
            correlation_df: DataFrame with correlation results
            
        Returns:
            Dictionary with RQ1 findings
        """
        # Look for AST, DFG, and PDG GED metrics
        graph_metrics = correlation_df[
            correlation_df['metric'].str.contains('ast|dfg|pdg', case=False, na=False)
        ].copy()
        
        if len(graph_metrics) == 0:
            self.logger.warning("No graph metrics found for RQ1 analysis")
            return {'status': 'insufficient_data'}
        
        # Find best predictor
        best = graph_metrics.nlargest(1, 'abs_spearman').iloc[0]
        
        rq1_results = {
            'question': 'Which graph representation best predicts LLM repair difficulty?',
            'best_predictor': best['metric'],
            'correlation': float(best['spearman_correlation']),
            'pvalue': float(best['spearman_pvalue']),
            'all_graph_metrics': graph_metrics[['metric', 'spearman_correlation', 'spearman_pvalue']].to_dict('records')
        }
        
        self.logger.info(f"RQ1: Best predictor is {best['metric']} (ρ={best['spearman_correlation']:.3f})")
        return rq1_results
    
    def answer_rq2(self, correlation_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Answer RQ2: Does semantic complexity (DFG/PDG) outperform syntactic complexity (AST/LOC)?
        
        Args:
            correlation_df: DataFrame with correlation results
            
        Returns:
            Dictionary with RQ2 findings
        """
        # Categorize metrics
        semantic = correlation_df[
            correlation_df['metric'].str.contains('dfg|pdg', case=False, na=False)
        ].copy()
        
        syntactic = correlation_df[
            correlation_df['metric'].str.contains('ast|loc', case=False, na=False)
        ].copy()
        
        if len(semantic) == 0 or len(syntactic) == 0:
            self.logger.warning("Insufficient metric data for RQ2 analysis")
            return {'status': 'insufficient_data'}
        
        # Compare average correlation strength
        semantic_avg = semantic['abs_spearman'].mean()
        syntactic_avg = syntactic['abs_spearman'].mean()
        
        rq2_results = {
            'question': 'Does semantic complexity outperform syntactic complexity?',
            'semantic_avg_correlation': float(semantic_avg),
            'syntactic_avg_correlation': float(syntactic_avg),
            'semantic_outperforms': semantic_avg > syntactic_avg,
            'difference': float(semantic_avg - syntactic_avg)
        }
        
        self.logger.info(f"RQ2: Semantic avg={semantic_avg:.3f}, Syntactic avg={syntactic_avg:.3f}")
        return rq2_results
    
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
