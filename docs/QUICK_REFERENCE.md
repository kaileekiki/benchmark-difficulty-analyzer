# Quick Reference: Multi-Difficulty Support

## CLI Commands

```bash
# Test with 100 bugs (fast iteration)
python main.py --limit 100

# Test with 50 bugs
python main.py --limit 50 --skip-crawl

# Full analysis
python main.py --benchmark swe_bench_verified

# Run comprehensive tests
python test_difficulty_support.py
```

## Python API

### Calculate Individual Difficulty Metrics

```python
from difficulty.metrics import DifficultyMetrics

# Method 1: Success rate-based
difficulty = DifficultyMetrics.from_model_success_rate(0.75)
# Output: 0.25 (25% difficulty)

# Method 2: Time-based
difficulty = DifficultyMetrics.from_time_difficulty("30 min")
# Output: 0.125 (12.5% of max 4 hours)

# Method 3: Tier-weighted
tier_results = {'90-100': 0.8, '80-90': 0.6, '70-80': 0.4}
difficulty = DifficultyMetrics.from_model_tiers(tier_results)
# Output: weighted difficulty based on tier performance

# Method 4: Combined
difficulty = DifficultyMetrics.combined(
    success_rate=0.7,
    time_difficulty=0.3,
    weights={'success_rate': 0.4, 'time': 0.6}
)
# Output: 0.300 (40% * (1-0.7) + 60% * 0.3)
```

### Load Time Difficulties from HuggingFace

```python
from data.loaders import SWEBenchVerifiedLoader

loader = SWEBenchVerifiedLoader()
all_difficulties = loader.get_all_difficulties()
# Returns: {'instance_id': '15 min', ...}

bug_difficulty = loader.get_difficulty_for_bug('django__django-12345')
# Returns: '30 min' or None
```

### Calculate Multiple Difficulties

```python
from analyzers.bug_resolver_analyzer import BugResolverAnalyzer

analyzer = BugResolverAnalyzer()
analyzer.create_bug_data_from_crawled(leaderboard_data, bug_results)

difficulty_config = {
    'weights': {'success_rate': 0.4, 'time': 0.6}
}

difficulty_data = analyzer.calculate_multiple_difficulties(
    leaderboard_data,
    difficulty_config
)
# Returns DataFrame with columns:
# - bug_id
# - resolution_rate
# - difficulty_success_rate
# - difficulty_time
# - difficulty_tier
# - difficulty_combined
# - time_difficulty_str
```

## Configuration

Edit `config/benchmarks.yaml`:

```yaml
swe_bench_verified:
  difficulty:
    methods:
      - success_rate      # Enable/disable methods
      - time
      - tier_weighted
      - combined
    primary: combined     # Primary method to use
    weights:
      success_rate: 0.4   # Adjust weights
      time: 0.6
```

## Output Format

The `model_bug_matrix.csv` will include:

```csv
model_name,bug1,bug2,bug3
Model 1,1,1,0
Model 2,1,0,0
difficulty_success_rate,0.330,0.670,1.000
difficulty_time,0.125,0.250,0.500
difficulty_tier,0.200,0.500,0.900
difficulty_combined,0.207,0.418,0.700
```

## Difficulty Score Interpretation

- **0.0 - 0.25**: Easy
- **0.25 - 0.50**: Medium
- **0.50 - 0.75**: Hard
- **0.75 - 1.0**: Very Hard

## Time String Formats

Supported:
- "15 min", "15 minutes"
- "30 min", "30 minutes"
- "1 hour", "2 hours"
- "45 min"

Normalized to [0, 1] with max = 4 hours (240 minutes)

## Error Handling

If HuggingFace dataset is unavailable:
- Loader logs warning
- Returns empty dict/None
- Time difficulty defaults to 0.5 (medium)
- Analysis continues with other methods

## Example Workflow

```python
# 1. Setup
from analyzers.bug_resolver_analyzer import BugResolverAnalyzer
from exporters.dataset_exporter import DatasetExporter

analyzer = BugResolverAnalyzer()
exporter = DatasetExporter()

# 2. Create bug data
bug_data = analyzer.create_bug_data_from_crawled(
    leaderboard_data,
    bug_results
)

# 3. Calculate difficulties
difficulty_data = analyzer.calculate_multiple_difficulties(
    leaderboard_data,
    {'weights': {'success_rate': 0.4, 'time': 0.6}}
)

# 4. Export with difficulty columns
exported_files = exporter.export_all_datasets(
    leaderboard_data,
    bug_data,
    difficulty_data
)

print(f"Exported to: {exported_files['model_bug_matrix']}")
```
