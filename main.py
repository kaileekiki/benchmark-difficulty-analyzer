#!/usr/bin/env python3
"""
Main orchestration script for benchmark difficulty analyzer.
Coordinates crawling, analysis, and visualization of benchmark data.
"""

import argparse
import sys
import os
import yaml
import logging
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logger
from crawlers import SWEBenchCrawler
from analyzers import (
    BugResolverAnalyzer,
    TierAnalyzer,
    DifficultyAnalyzer,
    ModelComparison
)
from visualizations import Plotter
from exporters import DatasetExporter


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main execution function."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Benchmark Difficulty Analyzer - Analyze bug difficulty in APR benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --benchmark verified --visualize
  python main.py --benchmark lite --skip-crawl --metrics-file data/metrics/bug_metrics.csv
  python main.py --benchmark verified --skip-crawl
        """
    )
    
    parser.add_argument(
        '--benchmark',
        type=str,
        default='swe_bench_verified',
        help='Benchmark to analyze (default: swe_bench_verified)'
    )
    
    parser.add_argument(
        '--skip-crawl',
        action='store_true',
        help='Skip crawling and use existing data'
    )
    
    parser.add_argument(
        '--metrics-file',
        type=str,
        help='Path to bug metrics CSV file (optional)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logger('main', log_level=log_level)
    
    logger.info("="*70)
    logger.info("BENCHMARK DIFFICULTY ANALYZER")
    logger.info("="*70)
    logger.info(f"Benchmark: {args.benchmark}")
    logger.info(f"Skip crawl: {args.skip_crawl}")
    logger.info(f"Visualize: {args.visualize}")
    logger.info("="*70)
    
    try:
        # Load configurations
        logger.info("Loading configuration files...")
        benchmark_config = load_config('config/benchmarks.yaml')
        
        if args.benchmark not in benchmark_config['benchmarks']:
            logger.error(f"Unknown benchmark: {args.benchmark}")
            logger.error(f"Available benchmarks: {list(benchmark_config['benchmarks'].keys())}")
            return 1
        
        selected_benchmark = benchmark_config['benchmarks'][args.benchmark]
        tier_ranges = benchmark_config['performance_tiers']['ranges']
        
        logger.info(f"Loaded config for: {selected_benchmark['name']}")
        
        # Step 1: Crawl leaderboard (unless skipped)
        leaderboard_data = None
        bug_results = {}
        
        if not args.skip_crawl:
            logger.info("\n" + "="*70)
            logger.info("STEP 1: CRAWLING LEADERBOARD")
            logger.info("="*70)
            
            crawler = SWEBenchCrawler(selected_benchmark)
            leaderboard_data = crawler.crawl_leaderboard()
            
            # Save leaderboard data
            crawler.save_data(leaderboard_data, 'leaderboard')
            
            logger.info(f"Crawled {len(leaderboard_data)} models from leaderboard")
            
            # Crawl bug-level results for each model
            logger.info("\nCrawling bug-level results...")
            for model_data in leaderboard_data:
                model_id = model_data['model_id']
                logger.info(f"  Crawling results for {model_id}...")
                bug_results[model_id] = crawler.crawl_bug_results(model_id)
            
            # Save bug results
            crawler.save_data(bug_results, 'bug_results')
            
        else:
            logger.info("\n" + "="*70)
            logger.info("STEP 1: LOADING EXISTING DATA")
            logger.info("="*70)
            
            crawler = SWEBenchCrawler(selected_benchmark)
            leaderboard_data = crawler.load_latest_data('leaderboard')
            bug_results = crawler.load_latest_data('bug_results')
            
            if leaderboard_data is None or bug_results is None:
                logger.error("No existing data found. Please run without --skip-crawl first.")
                return 1
            
            logger.info(f"Loaded data for {len(leaderboard_data)} models")
        
        # Step 2: Bug-level resolution analysis
        logger.info("\n" + "="*70)
        logger.info("STEP 2: BUG RESOLUTION ANALYSIS")
        logger.info("="*70)
        
        bug_analyzer = BugResolverAnalyzer()
        
        # Create bug data from crawled results
        bug_data = bug_analyzer.create_bug_data_from_crawled(leaderboard_data, bug_results)
        
        # Analyze resolution rates
        resolution_analysis = bug_analyzer.analyze_bug_resolution_rates()
        logger.info(f"Analyzed {len(resolution_analysis)} bugs")
        
        # Get difficulty distribution
        difficulty_dist = bug_analyzer.get_difficulty_distribution()
        logger.info(f"Difficulty distribution: {difficulty_dist}")
        
        # Identify hardest and easiest bugs
        hardest_bugs = bug_analyzer.identify_hardest_bugs(threshold=0.25)
        easiest_bugs = bug_analyzer.identify_easiest_bugs()
        logger.info(f"Found {len(hardest_bugs)} very hard bugs, {len(easiest_bugs)} easy bugs")
        
        # Analyze consensus
        consensus = bug_analyzer.analyze_consensus_bugs()
        logger.info(f"Consensus analysis: {consensus['all_pass'].sum()} all-pass, "
                   f"{consensus['all_fail'].sum()} all-fail bugs")
        
        # Save analysis
        bug_analyzer.save_analysis(resolution_analysis, 'bug_resolution_analysis')
        bug_analyzer.save_analysis(hardest_bugs, 'hardest_bugs')
        bug_analyzer.save_analysis(easiest_bugs, 'easiest_bugs')
        
        # Step 3: Tier analysis
        logger.info("\n" + "="*70)
        logger.info("STEP 3: PERFORMANCE TIER ANALYSIS")
        logger.info("="*70)
        
        tier_analyzer = TierAnalyzer(tier_ranges)
        
        # Assign models to tiers
        tier_assignments = tier_analyzer.assign_models_to_tiers(leaderboard_data)
        
        # Analyze tier overlap
        tier_stats = tier_analyzer.analyze_tier_overlap(bug_data, tier_assignments)
        logger.info(f"Analyzed {len(tier_stats)} performance tiers")
        
        # Calculate Jaccard similarity
        similarity = tier_analyzer.calculate_jaccard_similarity(bug_data, tier_assignments)
        logger.info(f"Calculated {len(similarity)} pairwise similarities")
        
        # Identify common bugs
        common_bugs = tier_analyzer.identify_common_bugs_by_tier(bug_data, tier_assignments)
        
        # Save analysis
        tier_analyzer.save_analysis(tier_stats, 'tier_statistics')
        tier_analyzer.save_analysis(similarity, 'tier_similarity')
        tier_analyzer.save_analysis(common_bugs, 'common_bugs_by_tier')
        
        # Step 4: Difficulty metric correlation (if metrics provided)
        if args.metrics_file:
            logger.info("\n" + "="*70)
            logger.info("STEP 4: DIFFICULTY METRIC CORRELATION")
            logger.info("="*70)
            
            difficulty_analyzer = DifficultyAnalyzer()
            
            # Load metrics
            metrics_data = difficulty_analyzer.load_metrics(args.metrics_file)
            
            # Merge with resolution data
            merged_data = difficulty_analyzer.merge_with_resolution_data(resolution_analysis)
            
            # Calculate correlations
            correlations = difficulty_analyzer.calculate_correlations(merged_data)
            logger.info(f"Calculated correlations for {len(correlations)} metrics")
            
            # Analyze by difficulty category
            category_stats = difficulty_analyzer.analyze_by_difficulty_category(merged_data)
            
            # Identify top predictors
            top_predictors = difficulty_analyzer.identify_top_predictors(correlations)
            logger.info(f"Top {len(top_predictors)} difficulty predictors identified")
            
            # Answer research questions
            rq1 = difficulty_analyzer.answer_rq1(correlations)
            rq2 = difficulty_analyzer.answer_rq2(correlations)
            
            logger.info(f"\nRQ1: {rq1.get('question', 'N/A')}")
            logger.info(f"  Best predictor: {rq1.get('best_predictor', 'N/A')} "
                       f"(ρ={rq1.get('correlation', 0):.3f})")
            
            logger.info(f"\nRQ2: {rq2.get('question', 'N/A')}")
            logger.info(f"  Semantic outperforms: {rq2.get('semantic_outperforms', False)}")
            
            # Save analysis
            difficulty_analyzer.save_analysis(correlations, 'metric_correlations')
            difficulty_analyzer.save_analysis(category_stats, 'difficulty_category_stats')
        else:
            logger.info("\n" + "="*70)
            logger.info("STEP 4: SKIPPED (no metrics file provided)")
            logger.info("="*70)
            correlations = None
        
        # Step 5: Model comparison
        logger.info("\n" + "="*70)
        logger.info("STEP 5: MODEL COMPARISON ANALYSIS")
        logger.info("="*70)
        
        model_comparison = ModelComparison()
        
        # Calculate pairwise agreement
        agreement = model_comparison.calculate_pairwise_agreement(bug_data)
        logger.info(f"Calculated agreement for {len(agreement)} model pairs")
        
        # Find model clusters
        clusters = model_comparison.find_model_clusters(agreement, threshold=0.8)
        logger.info(f"Found {len(clusters)} model clusters with high agreement")
        
        # Generate performance matrix
        performance_matrix = model_comparison.generate_performance_matrix(bug_data)
        logger.info(f"Generated performance matrix: {performance_matrix.shape}")
        
        # Identify unique solvers
        unique_solvers = model_comparison.identify_unique_solvers(bug_data)
        logger.info(f"Found {len(unique_solvers)} bugs with unique solvers")
        
        # Save analysis
        model_comparison.save_analysis(agreement, 'model_agreement')
        model_comparison.save_analysis(unique_solvers, 'unique_solvers')
        
        # Step 6: Export Datasets
        logger.info("\n" + "="*70)
        logger.info("STEP 6: EXPORTING DATASETS")
        logger.info("="*70)
        
        exporter = DatasetExporter()
        exported_files = exporter.export_all_datasets(leaderboard_data, bug_data)
        
        logger.info("\nExported datasets:")
        for dataset_name, file_path in exported_files.items():
            logger.info(f"  {dataset_name}: {file_path}")
        
        # Step 7: Visualization (if requested)
        if args.visualize:
            logger.info("\n" + "="*70)
            logger.info("STEP 7: GENERATING VISUALIZATIONS")
            logger.info("="*70)
            
            plotter = Plotter()
            
            # Plot bug difficulty distribution
            logger.info("Creating bug difficulty distribution plot...")
            plotter.plot_bug_difficulty_distribution(resolution_analysis)
            
            # Plot tier performance
            logger.info("Creating tier performance plot...")
            plotter.plot_tier_performance(tier_stats)
            
            # Plot metric correlations (if available)
            if correlations is not None:
                logger.info("Creating metric correlations plot...")
                plotter.plot_metric_correlations(correlations)
            
            # Plot model agreement heatmap
            logger.info("Creating model agreement heatmap...")
            plotter.plot_model_agreement_heatmap(agreement)
            
            # Plot performance matrix
            logger.info("Creating performance matrix plot...")
            plotter.plot_performance_matrix(performance_matrix)
            
            logger.info(f"\nAll visualizations saved to visualizations/output/")
        else:
            logger.info("\n" + "="*70)
            logger.info("STEP 7: SKIPPED (use --visualize to generate plots)")
            logger.info("="*70)
        
        # Step 8: Summary statistics
        logger.info("\n" + "="*70)
        logger.info("SUMMARY STATISTICS")
        logger.info("="*70)
        
        total_bugs = len(resolution_analysis)
        total_models = len(leaderboard_data)
        
        summary_stats = {
            'total_bugs': total_bugs,
            'total_models': total_models,
            'total_evaluations': len(bug_data),
            'easy_bugs': difficulty_dist.get('Easy', 0),
            'medium_bugs': difficulty_dist.get('Medium', 0),
            'hard_bugs': difficulty_dist.get('Hard', 0),
            'very_hard_bugs': difficulty_dist.get('Very Hard', 0),
            'easy_pct': difficulty_dist.get('Easy', 0) / total_bugs * 100,
            'medium_pct': difficulty_dist.get('Medium', 0) / total_bugs * 100,
            'hard_pct': difficulty_dist.get('Hard', 0) / total_bugs * 100,
            'very_hard_pct': difficulty_dist.get('Very Hard', 0) / total_bugs * 100,
            'mean_resolution_rate': resolution_analysis['resolution_rate'].mean(),
            'median_resolution_rate': resolution_analysis['resolution_rate'].median(),
            'hardest_bug': resolution_analysis.iloc[0]['bug_id'],
            'hardest_rate': resolution_analysis.iloc[0]['resolution_rate'],
            'easiest_bug': resolution_analysis.iloc[-1]['bug_id'],
            'easiest_rate': resolution_analysis.iloc[-1]['resolution_rate']
        }
        
        # Add top metrics if available
        if correlations is not None:
            top_3_metrics = []
            for _, row in correlations.head(3).iterrows():
                top_3_metrics.append({
                    'metric': row['metric'],
                    'correlation': row['spearman_correlation']
                })
            summary_stats['top_metrics'] = top_3_metrics
        
        # Print and save summary
        if args.visualize:
            plotter = Plotter()
            plotter.create_summary_report(summary_stats)
        else:
            # Just print summary stats
            print(f"\nTotal Bugs: {total_bugs}")
            print(f"Total Models: {total_models}")
            print(f"Total Evaluations: {len(bug_data)}")
            print(f"\nDifficulty Distribution:")
            print(f"  Easy: {difficulty_dist.get('Easy', 0)} ({summary_stats['easy_pct']:.1f}%)")
            print(f"  Medium: {difficulty_dist.get('Medium', 0)} ({summary_stats['medium_pct']:.1f}%)")
            print(f"  Hard: {difficulty_dist.get('Hard', 0)} ({summary_stats['hard_pct']:.1f}%)")
            print(f"  Very Hard: {difficulty_dist.get('Very Hard', 0)} ({summary_stats['very_hard_pct']:.1f}%)")
        
        logger.info("\n" + "="*70)
        logger.info("ANALYSIS COMPLETE!")
        logger.info("="*70)
        logger.info(f"Results saved to: data/processed/")
        logger.info(f"\nExported datasets:")
        for dataset_name, file_path in exported_files.items():
            logger.info(f"  - {os.path.basename(file_path)}")
        if args.visualize:
            logger.info(f"Visualizations saved to: visualizations/output/")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
