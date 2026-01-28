"""
Visualization and plotting functions for benchmark analysis.
Creates publication-quality plots for difficulty analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime


# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class Plotter:
    """
    Creates visualizations for benchmark difficulty analysis.
    """
    
    def __init__(self, output_dir: str = "visualizations/output"):
        """
        Initialize the plotter.
        
        Args:
            output_dir: Directory to save plot images
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_bug_difficulty_distribution(self, resolution_data: pd.DataFrame, 
                                        save_name: str = "bug_difficulty_distribution"):
        """
        Plot bug difficulty distribution as histogram and category bar chart.
        
        Args:
            resolution_data: DataFrame with bug_id, resolution_rate, difficulty_label
            save_name: Base name for saved plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of resolution rates
        axes[0].hist(resolution_data['resolution_rate'], bins=20, 
                    edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Resolution Rate')
        axes[0].set_ylabel('Number of Bugs')
        axes[0].set_title('Distribution of Bug Resolution Rates')
        axes[0].axvline(resolution_data['resolution_rate'].mean(), 
                       color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {resolution_data["resolution_rate"].mean():.2f}')
        axes[0].legend()
        
        # Bar chart of difficulty categories
        difficulty_counts = resolution_data['difficulty_label'].value_counts()
        difficulty_order = ['Easy', 'Medium', 'Hard', 'Very Hard']
        difficulty_counts = difficulty_counts.reindex(difficulty_order, fill_value=0)
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#8b0000']
        axes[1].bar(range(len(difficulty_counts)), difficulty_counts.values, 
                   color=colors, edgecolor='black', alpha=0.8)
        axes[1].set_xticks(range(len(difficulty_counts)))
        axes[1].set_xticklabels(difficulty_counts.index, rotation=0)
        axes[1].set_xlabel('Difficulty Category')
        axes[1].set_ylabel('Number of Bugs')
        axes[1].set_title('Bug Count by Difficulty Category')
        
        # Add count labels on bars
        for i, v in enumerate(difficulty_counts.values):
            axes[1].text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved bug difficulty distribution plot to {filepath}")
    
    def plot_tier_performance(self, tier_stats: pd.DataFrame,
                             save_name: str = "tier_performance"):
        """
        Plot 4-panel tier analysis visualization.
        
        Args:
            tier_stats: DataFrame with tier statistics
            save_name: Base name for saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Sort tiers for consistent ordering
        tier_stats = tier_stats.sort_values('tier')
        
        # Panel 1: Models per tier
        axes[0, 0].bar(range(len(tier_stats)), tier_stats['num_models'], 
                       color='skyblue', edgecolor='black', alpha=0.8)
        axes[0, 0].set_xticks(range(len(tier_stats)))
        axes[0, 0].set_xticklabels(tier_stats['tier'], rotation=45, ha='right')
        axes[0, 0].set_ylabel('Number of Models')
        axes[0, 0].set_title('Models per Performance Tier')
        
        # Panel 2: Resolution rate by tier
        axes[0, 1].plot(range(len(tier_stats)), tier_stats['resolution_rate'], 
                       marker='o', linewidth=2, markersize=8, color='green')
        axes[0, 1].set_xticks(range(len(tier_stats)))
        axes[0, 1].set_xticklabels(tier_stats['tier'], rotation=45, ha='right')
        axes[0, 1].set_ylabel('Resolution Rate')
        axes[0, 1].set_title('Resolution Rate by Tier')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(True, alpha=0.3)
        
        # Panel 3: Total attempts by tier
        axes[1, 0].bar(range(len(tier_stats)), tier_stats['total_attempts'], 
                       color='coral', edgecolor='black', alpha=0.8)
        axes[1, 0].set_xticks(range(len(tier_stats)))
        axes[1, 0].set_xticklabels(tier_stats['tier'], rotation=45, ha='right')
        axes[1, 0].set_ylabel('Total Attempts')
        axes[1, 0].set_title('Bug Resolution Attempts by Tier')
        
        # Panel 4: Unique bugs by tier
        axes[1, 1].bar(range(len(tier_stats)), tier_stats['unique_bugs'], 
                       color='mediumpurple', edgecolor='black', alpha=0.8)
        axes[1, 1].set_xticks(range(len(tier_stats)))
        axes[1, 1].set_xticklabels(tier_stats['tier'], rotation=45, ha='right')
        axes[1, 1].set_ylabel('Unique Bugs')
        axes[1, 1].set_title('Unique Bugs Attempted by Tier')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved tier performance plot to {filepath}")
    
    def plot_metric_correlations(self, correlation_df: pd.DataFrame,
                                 top_n: int = 10,
                                 save_name: str = "metric_correlations"):
        """
        Plot top metric correlations with difficulty.
        
        Args:
            correlation_df: DataFrame with correlation results
            top_n: Number of top metrics to plot
            save_name: Base name for saved plot
        """
        # Get top N metrics by absolute correlation
        top_metrics = correlation_df.nlargest(top_n, 'abs_spearman')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bar plot
        colors = ['green' if x > 0 else 'red' for x in top_metrics['spearman_correlation']]
        bars = ax.barh(range(len(top_metrics)), top_metrics['spearman_correlation'], 
                       color=colors, alpha=0.7, edgecolor='black')
        
        # Customize plot
        ax.set_yticks(range(len(top_metrics)))
        ax.set_yticklabels(top_metrics['metric'])
        ax.set_xlabel('Spearman Correlation with Resolution Rate')
        ax.set_title(f'Top {top_n} Predictive Metrics for Bug Difficulty')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add correlation values as text
        for i, (corr, pval) in enumerate(zip(top_metrics['spearman_correlation'], 
                                             top_metrics['spearman_pvalue'])):
            significance = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
            ax.text(corr, i, f' {corr:.3f}{significance}', va='center', 
                   ha='left' if corr > 0 else 'right', fontsize=9)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved metric correlations plot to {filepath}")
    
    def plot_model_agreement_heatmap(self, agreement_df: pd.DataFrame,
                                    save_name: str = "model_agreement_heatmap"):
        """
        Plot pairwise model agreement as heatmap.
        
        Args:
            agreement_df: DataFrame with pairwise agreement scores
            save_name: Base name for saved plot
        """
        # Create pivot table for heatmap
        models = sorted(set(agreement_df['model1'].unique()) | set(agreement_df['model2'].unique()))
        
        # Initialize matrix
        agreement_matrix = pd.DataFrame(
            np.zeros((len(models), len(models))),
            index=models,
            columns=models
        )
        
        # Fill matrix
        for _, row in agreement_df.iterrows():
            agreement_matrix.loc[row['model1'], row['model2']] = row['agreement_rate']
            agreement_matrix.loc[row['model2'], row['model1']] = row['agreement_rate']
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(agreement_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                   vmin=0, vmax=1, square=True, linewidths=0.5,
                   cbar_kws={'label': 'Agreement Rate'}, ax=ax)
        
        ax.set_title('Pairwise Model Agreement Heatmap', fontsize=14, pad=20)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        
        # Rotate labels for readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved model agreement heatmap to {filepath}")
    
    def plot_performance_matrix(self, performance_matrix: pd.DataFrame,
                               max_bugs: int = 50,
                               save_name: str = "performance_matrix"):
        """
        Plot bugs × models performance matrix as heatmap.
        
        Args:
            performance_matrix: DataFrame with bugs as rows, models as columns
            max_bugs: Maximum number of bugs to display (sample if exceeded)
            save_name: Base name for saved plot
        """
        # Sample bugs if too many
        if len(performance_matrix) > max_bugs:
            self.logger.info(f"Sampling {max_bugs} bugs from {len(performance_matrix)} total")
            # Sample diverse bugs (some easy, some hard)
            bug_difficulty = performance_matrix.mean(axis=1).sort_values()
            sample_indices = np.linspace(0, len(bug_difficulty)-1, max_bugs, dtype=int)
            sampled_bugs = bug_difficulty.iloc[sample_indices].index
            performance_matrix = performance_matrix.loc[sampled_bugs]
        
        fig, ax = plt.subplots(figsize=(14, max(8, len(performance_matrix) * 0.15)))
        
        sns.heatmap(performance_matrix, cmap=['#ffcccc', '#ccffcc'],
                   cbar_kws={'label': 'Resolution Status', 'ticks': [0.25, 0.75]},
                   linewidths=0.1, linecolor='gray', ax=ax)
        
        # Customize colorbar labels
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticklabels(['Failed', 'Solved'])
        
        ax.set_title('Bug Resolution Matrix (Bugs × Models)', fontsize=14, pad=20)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Bug ID', fontsize=12)
        
        # Rotate labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(fontsize=7)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f"{save_name}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved performance matrix to {filepath}")
    
    def create_summary_report(self, stats: Dict[str, Any],
                             save_name: str = "analysis_summary"):
        """
        Create a text summary report with key statistics.
        
        Args:
            stats: Dictionary with summary statistics
            save_name: Base name for saved report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║     BENCHMARK DIFFICULTY ANALYSIS - SUMMARY REPORT          ║
╚══════════════════════════════════════════════════════════════╝

Generated: {timestamp}

─────────────────────────────────────────────────────────────
DATASET OVERVIEW
─────────────────────────────────────────────────────────────
Total Bugs:           {stats.get('total_bugs', 'N/A')}
Total Models:         {stats.get('total_models', 'N/A')}
Total Evaluations:    {stats.get('total_evaluations', 'N/A')}

─────────────────────────────────────────────────────────────
DIFFICULTY DISTRIBUTION
─────────────────────────────────────────────────────────────
Easy Bugs:            {stats.get('easy_bugs', 'N/A')} ({stats.get('easy_pct', 0):.1f}%)
Medium Bugs:          {stats.get('medium_bugs', 'N/A')} ({stats.get('medium_pct', 0):.1f}%)
Hard Bugs:            {stats.get('hard_bugs', 'N/A')} ({stats.get('hard_pct', 0):.1f}%)
Very Hard Bugs:       {stats.get('very_hard_bugs', 'N/A')} ({stats.get('very_hard_pct', 0):.1f}%)

─────────────────────────────────────────────────────────────
KEY FINDINGS
─────────────────────────────────────────────────────────────
Mean Resolution Rate: {stats.get('mean_resolution_rate', 'N/A'):.2%}
Median Resolution:    {stats.get('median_resolution_rate', 'N/A'):.2%}

Hardest Bug:          {stats.get('hardest_bug', 'N/A')} (rate: {stats.get('hardest_rate', 0):.2%})
Easiest Bug:          {stats.get('easiest_bug', 'N/A')} (rate: {stats.get('easiest_rate', 0):.2%})

─────────────────────────────────────────────────────────────
METRIC CORRELATIONS (Top 3)
─────────────────────────────────────────────────────────────
"""
        
        if 'top_metrics' in stats:
            for i, metric in enumerate(stats['top_metrics'][:3], 1):
                report += f"{i}. {metric.get('metric', 'N/A'):30s} ρ = {metric.get('correlation', 0):6.3f}\n"
        else:
            report += "No correlation data available\n"
        
        report += "\n"
        report += "─────────────────────────────────────────────────────────────\n"
        report += "END OF REPORT\n"
        report += "─────────────────────────────────────────────────────────────\n"
        
        # Save report
        filepath = os.path.join(self.output_dir, f"{save_name}.txt")
        with open(filepath, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Saved summary report to {filepath}")
        
        # Also print to console
        print(report)


# Convenience functions for backward compatibility
def plot_bug_difficulty_distribution(resolution_data: pd.DataFrame, 
                                    output_dir: str = "visualizations/output",
                                    save_name: str = "bug_difficulty_distribution"):
    """Convenience function for plotting bug difficulty distribution."""
    plotter = Plotter(output_dir)
    plotter.plot_bug_difficulty_distribution(resolution_data, save_name)


def plot_tier_performance(tier_stats: pd.DataFrame,
                         output_dir: str = "visualizations/output",
                         save_name: str = "tier_performance"):
    """Convenience function for plotting tier performance."""
    plotter = Plotter(output_dir)
    plotter.plot_tier_performance(tier_stats, save_name)


def plot_metric_correlations(correlation_df: pd.DataFrame,
                             top_n: int = 10,
                             output_dir: str = "visualizations/output",
                             save_name: str = "metric_correlations"):
    """Convenience function for plotting metric correlations."""
    plotter = Plotter(output_dir)
    plotter.plot_metric_correlations(correlation_df, top_n, save_name)


def plot_model_agreement_heatmap(agreement_df: pd.DataFrame,
                                output_dir: str = "visualizations/output",
                                save_name: str = "model_agreement_heatmap"):
    """Convenience function for plotting model agreement heatmap."""
    plotter = Plotter(output_dir)
    plotter.plot_model_agreement_heatmap(agreement_df, save_name)


def plot_performance_matrix(performance_matrix: pd.DataFrame,
                           max_bugs: int = 50,
                           output_dir: str = "visualizations/output",
                           save_name: str = "performance_matrix"):
    """Convenience function for plotting performance matrix."""
    plotter = Plotter(output_dir)
    plotter.plot_performance_matrix(performance_matrix, max_bugs, save_name)
