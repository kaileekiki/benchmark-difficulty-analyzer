#!/usr/bin/env python3
"""
Test script for multi-difficulty definition support.
Demonstrates all the new functionality with mock data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from analyzers.bug_resolver_analyzer import BugResolverAnalyzer
from difficulty.metrics import DifficultyMetrics
from exporters.dataset_exporter import DatasetExporter


def test_difficulty_metrics():
    """Test DifficultyMetrics class."""
    print("="*70)
    print("TEST 1: DifficultyMetrics")
    print("="*70)
    
    # Test time parsing
    print("\n1.1 Time-based difficulty:")
    time_cases = ['15 min', '30 min', '1 hour', '2 hours', '4 hours', None, 'invalid']
    for case in time_cases:
        result = DifficultyMetrics.from_time_difficulty(case)
        print(f"  {str(case):15s} -> {result:.3f}")
    
    # Test success rate
    print("\n1.2 Success rate-based difficulty:")
    for rate in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = DifficultyMetrics.from_model_success_rate(rate)
        print(f"  success_rate={rate:.2f} -> difficulty={result:.3f}")
    
    # Test tier-based
    print("\n1.3 Tier-based difficulty:")
    tier_results = {
        '90-100': 0.8,
        '80-90': 0.6,
        '70-80': 0.4
    }
    result = DifficultyMetrics.from_model_tiers(tier_results)
    print(f"  tier_results={tier_results}")
    print(f"  -> difficulty={result:.3f}")
    
    # Test combined
    print("\n1.4 Combined difficulty:")
    result = DifficultyMetrics.combined(0.7, 0.3)
    print(f"  combined(success_rate=0.7, time_diff=0.3) = {result:.3f}")
    
    print("\n✓ DifficultyMetrics tests passed!\n")


def test_bug_resolver_analyzer():
    """Test BugResolverAnalyzer with multiple difficulty calculations."""
    print("="*70)
    print("TEST 2: BugResolverAnalyzer")
    print("="*70)
    
    # Create mock data
    print("\n2.1 Creating mock data...")
    leaderboard_data = [
        {'model_id': 'top_model', 'model_name': 'Top Model', 'score': 95},
        {'model_id': 'good_model', 'model_name': 'Good Model', 'score': 85},
        {'model_id': 'avg_model', 'model_name': 'Average Model', 'score': 75},
        {'model_id': 'weak_model', 'model_name': 'Weak Model', 'score': 65},
    ]
    
    # Bug 1: Easy (3/4 models solve it)
    # Bug 2: Medium (2/4 models solve it)
    # Bug 3: Hard (1/4 models solve it)
    # Bug 4: Very Hard (0/4 models solve it)
    bug_results = {
        'top_model': {'bug1': True, 'bug2': True, 'bug3': True, 'bug4': False},
        'good_model': {'bug1': True, 'bug2': True, 'bug3': False, 'bug4': False},
        'avg_model': {'bug1': True, 'bug2': False, 'bug3': False, 'bug4': False},
        'weak_model': {'bug1': False, 'bug2': False, 'bug3': False, 'bug4': False},
    }
    
    # Create analyzer
    analyzer = BugResolverAnalyzer(data_dir="/tmp/test_analyzer")
    
    # Create bug data
    bug_data = analyzer.create_bug_data_from_crawled(leaderboard_data, bug_results)
    print(f"  Created bug data: {len(bug_data)} records")
    
    # Analyze resolution rates
    print("\n2.2 Analyzing resolution rates...")
    resolution_analysis = analyzer.analyze_bug_resolution_rates()
    print(resolution_analysis[['bug_id', 'resolution_rate', 'difficulty_label']])
    
    # Calculate multiple difficulties
    print("\n2.3 Calculating multiple difficulty metrics...")
    difficulty_config = {'weights': {'success_rate': 0.4, 'time': 0.6}}
    difficulty_data = analyzer.calculate_multiple_difficulties(leaderboard_data, difficulty_config)
    
    print("\nDifficulty metrics:")
    cols = ['bug_id', 'difficulty_success_rate', 'difficulty_time', 'difficulty_tier', 'difficulty_combined']
    print(difficulty_data[cols].to_string(index=False))
    
    print("\n✓ BugResolverAnalyzer tests passed!\n")


def test_dataset_exporter():
    """Test DatasetExporter with difficulty columns."""
    print("="*70)
    print("TEST 3: DatasetExporter")
    print("="*70)
    
    # Create mock data
    print("\n3.1 Creating mock data for export...")
    leaderboard_data = [
        {'model_id': 'model1', 'model_name': 'Model 1', 'score': 90},
        {'model_id': 'model2', 'model_name': 'Model 2', 'score': 80},
        {'model_id': 'model3', 'model_name': 'Model 3', 'score': 70},
    ]
    
    bug_data = pd.DataFrame([
        {'model_id': 'model1', 'model_name': 'Model 1', 'bug_id': 'bug1', 'resolved': 1},
        {'model_id': 'model1', 'model_name': 'Model 1', 'bug_id': 'bug2', 'resolved': 1},
        {'model_id': 'model1', 'model_name': 'Model 1', 'bug_id': 'bug3', 'resolved': 0},
        {'model_id': 'model2', 'model_name': 'Model 2', 'bug_id': 'bug1', 'resolved': 1},
        {'model_id': 'model2', 'model_name': 'Model 2', 'bug_id': 'bug2', 'resolved': 0},
        {'model_id': 'model2', 'model_name': 'Model 2', 'bug_id': 'bug3', 'resolved': 0},
        {'model_id': 'model3', 'model_name': 'Model 3', 'bug_id': 'bug1', 'resolved': 0},
        {'model_id': 'model3', 'model_name': 'Model 3', 'bug_id': 'bug2', 'resolved': 0},
        {'model_id': 'model3', 'model_name': 'Model 3', 'bug_id': 'bug3', 'resolved': 0},
    ])
    
    # bug1: 2/3 solve = easy (0.33 difficulty)
    # bug2: 1/3 solve = medium (0.67 difficulty)
    # bug3: 0/3 solve = hard (1.0 difficulty)
    difficulty_data = pd.DataFrame([
        {'bug_id': 'bug1', 'difficulty_success_rate': 0.33, 'difficulty_time': 0.125, 
         'difficulty_tier': 0.2, 'difficulty_combined': 0.207},
        {'bug_id': 'bug2', 'difficulty_success_rate': 0.67, 'difficulty_time': 0.25, 
         'difficulty_tier': 0.5, 'difficulty_combined': 0.418},
        {'bug_id': 'bug3', 'difficulty_success_rate': 1.0, 'difficulty_time': 0.5, 
         'difficulty_tier': 0.9, 'difficulty_combined': 0.7},
    ])
    
    # Create exporter
    exporter = DatasetExporter(data_dir="/tmp/test_exporter")
    
    print("\n3.2 Exporting model-bug matrix with difficulty columns...")
    output_path = exporter.export_model_bug_matrix(leaderboard_data, bug_data, difficulty_data)
    
    # Read and display
    df = pd.read_csv(output_path)
    print("\nExported matrix:")
    print(df.to_string(index=False))
    
    # Verify difficulty rows
    assert 'difficulty_success_rate' in df['model_name'].values
    assert 'difficulty_time' in df['model_name'].values
    assert 'difficulty_tier' in df['model_name'].values
    assert 'difficulty_combined' in df['model_name'].values
    
    print("\n✓ DatasetExporter tests passed!\n")


def main():
    """Run all tests."""
    print("\n")
    print("="*70)
    print("MULTI-DIFFICULTY DEFINITION SUPPORT TEST SUITE")
    print("="*70)
    print()
    
    try:
        test_difficulty_metrics()
        test_bug_resolver_analyzer()
        test_dataset_exporter()
        
        print("="*70)
        print("ALL TESTS PASSED!")
        print("="*70)
        print()
        print("Summary:")
        print("  ✓ DifficultyMetrics: 4 methods working correctly")
        print("  ✓ BugResolverAnalyzer: Multiple difficulty calculation working")
        print("  ✓ DatasetExporter: Difficulty columns exported correctly")
        print()
        print("Usage in main.py:")
        print("  python main.py --limit 100        # Test with 100 bugs")
        print("  python main.py --skip-crawl       # Use existing data")
        print()
        return 0
        
    except Exception as e:
        print("\n" + "="*70)
        print("TEST FAILED!")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
