# Multi-Difficulty Definition Support

This document describes the new multi-difficulty definition support added to the benchmark-difficulty-analyzer.

## Overview

Previously, the analyzer used only a single difficulty metric based on model success rates (R² = 12%). This update adds support for multiple difficulty calculation methods to improve prediction performance.

## Features

### 1. Multiple Difficulty Metrics

Four different difficulty calculation methods are now supported:

#### 1.1 Success Rate-Based (Original)
```python
difficulty = 1.0 - success_rate
```
- **Range**: 0.0 (easiest) to 1.0 (hardest)
- **Based on**: Proportion of models that successfully resolved the bug

#### 1.2 Time-Based
```python
difficulty = DifficultyMetrics.from_time_difficulty("30 min")
```
- **Range**: 0.0 to 1.0 (normalized, max 4 hours)
- **Based on**: SWE-bench Verified's `difficulty` field
- **Supported formats**: "15 min", "30 min", "1 hour", "2 hours", etc.

#### 1.3 Tier-Weighted
```python
difficulty = DifficultyMetrics.from_model_tiers(tier_results)
```
- **Range**: 0.0 to 1.0
- **Based on**: Weighted success rates across model performance tiers
- **Logic**: Higher-tier models get more weight; if top models fail, bug is harder

#### 1.4 Combined
```python
difficulty = DifficultyMetrics.combined(success_rate, time_difficulty)
```
- **Range**: 0.0 to 1.0
- **Default weights**: 40% success rate, 60% time-based
- **Configurable**: Weights can be adjusted in `config/benchmarks.yaml`

### 2. Data Loader for SWE-bench Verified

A new loader fetches time-based difficulty from HuggingFace:

```python
from data.loaders import SWEBenchVerifiedLoader

loader = SWEBenchVerifiedLoader()
difficulties = loader.get_all_difficulties()
# Returns: {'bug_id': '15 min', ...}
```

### 3. Enhanced Dataset Export

The `model_bug_matrix.csv` now includes 4 additional rows with difficulty metrics:

```csv
model_name,bug1,bug2,bug3,...
Model 1,1,1,0,...
Model 2,1,0,0,...
difficulty_success_rate,0.330,0.670,1.0,...
difficulty_time,0.125,0.250,0.5,...
difficulty_tier,0.200,0.500,0.9,...
difficulty_combined,0.207,0.418,0.7,...
```

### 4. Test Mode with --limit

For rapid iteration and testing:

```bash
# Test with first 100 bugs
python main.py --limit 100

# Test with first 50 bugs
python main.py --limit 50 --skip-crawl
```

## Usage

### Basic Usage

```bash
# Full analysis with multiple difficulty metrics
python main.py --benchmark swe_bench_verified

# Test mode (100 bugs only)
python main.py --limit 100

# Use existing data
python main.py --skip-crawl
```

### Configuration

Edit `config/benchmarks.yaml` to customize difficulty calculation:

```yaml
swe_bench_verified:
  difficulty:
    methods:
      - success_rate
      - time
      - tier_weighted
      - combined
    primary: combined
    weights:
      success_rate: 0.4  # Adjust these weights
      time: 0.6
```

### Programmatic Usage

```python
from difficulty.metrics import DifficultyMetrics

# Calculate different difficulty metrics
success_rate = 0.75
time_str = "30 min"

# Method 1: Success rate-based
diff1 = DifficultyMetrics.from_model_success_rate(success_rate)

# Method 2: Time-based
diff2 = DifficultyMetrics.from_time_difficulty(time_str)

# Method 3: Tier-weighted
tier_results = {'90-100': 0.8, '80-90': 0.6}
diff3 = DifficultyMetrics.from_model_tiers(tier_results)

# Method 4: Combined (40% success, 60% time)
diff4 = DifficultyMetrics.combined(success_rate, diff2)
```

## Files Modified/Created

### New Files
- `data/loaders/swebench_verified_loader.py` - Loads difficulty from HuggingFace
- `data/loaders/__init__.py` - Loader module
- `difficulty/metrics.py` - Multiple difficulty calculation methods
- `difficulty/__init__.py` - Metrics module
- `test_difficulty_support.py` - Comprehensive test suite

### Modified Files
- `analyzers/bug_resolver_analyzer.py` - Added `calculate_multiple_difficulties()`
- `exporters/dataset_exporter.py` - Added difficulty columns to exports
- `main.py` - Added `--limit` option and difficulty calculation
- `config/benchmarks.yaml` - Added difficulty configuration
- `requirements.txt` - Added `datasets>=2.14.0`

## Testing

Run the test suite to verify the implementation:

```bash
python test_difficulty_support.py
```

Expected output:
```
======================================================================
ALL TESTS PASSED!
======================================================================

Summary:
  ✓ DifficultyMetrics: 4 methods working correctly
  ✓ BugResolverAnalyzer: Multiple difficulty calculation working
  ✓ DatasetExporter: Difficulty columns exported correctly
```

## Implementation Details

### Time Parsing Logic

The time parser handles various formats:
- "15 min", "30 minutes" → minutes
- "1 hour", "2 hours" → converted to minutes
- Maximum time: 4 hours (240 minutes)
- Normalization: `difficulty = min(minutes / 240.0, 1.0)`

### Tier Weighting

Tiers are weighted by their midpoint score:
- Tier 90-100: weight = 0.95
- Tier 80-90: weight = 0.85
- ...and so on

The difficulty is calculated as:
```python
difficulty = 1.0 - (weighted_sum_of_success_rates / total_weight)
```

### Combined Difficulty

The combined method uses weighted average:
```python
combined = (w1 * success_difficulty) + (w2 * time_difficulty)
```

Default weights can be customized in the config file.

## Benefits

1. **Improved Prediction**: Multiple signals instead of single metric
2. **Flexibility**: Choose which difficulty definition to use
3. **Interpretability**: Different perspectives on bug difficulty
4. **Extensibility**: Easy to add new difficulty calculation methods

## Future Work

Potential enhancements:
- Add ML-based difficulty prediction
- Include code complexity metrics (AST-GED, DFG-GED)
- Automatic weight optimization
- Difficulty prediction before bug resolution
