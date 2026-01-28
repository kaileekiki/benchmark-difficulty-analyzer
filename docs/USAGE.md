# Comprehensive Usage Guide for Benchmark Difficulty Analyzer

## Table of Contents
1. [Quick Start](#quick-start)
2. [Command-Line Options](#command-line-options)
3. [Expected Data Formats](#expected-data-formats)
4. [Extending to Other Benchmarks](#extending-to-other-benchmarks)
5. [Output Files](#output-files)
6. [Research Questions Mapping](#research-questions-mapping)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)
9. [Next Steps](#next-steps)

## Quick Start

### Installation

1. Install Python 3.8 or higher
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

### Basic Usage

Run complete analysis with visualizations:

```bash
python main.py --benchmark verified --visualize
```

This will:
1. Crawl the SWE-bench Verified leaderboard
2. Analyze bug resolution rates
3. Perform tier analysis
4. Compare models
5. Generate all visualizations

## Command-Line Options

### Required Arguments

None - all arguments have sensible defaults.

### Optional Arguments

#### `--benchmark <name>`
- **Description**: Benchmark to analyze
- **Options**: `verified`, `lite`, or any benchmark defined in `config/benchmarks.yaml`
- **Default**: `swe_bench_verified`
- **Example**: `--benchmark lite`

#### `--skip-crawl`
- **Description**: Skip web crawling and use existing data files
- **Use when**: You've already crawled data and want to re-run analysis
- **Example**: `--skip-crawl`

#### `--metrics-file <path>`
- **Description**: Path to CSV file containing bug metrics
- **Required for**: Metric correlation analysis (RQ1, RQ2)
- **Example**: `--metrics-file data/metrics/bug_metrics.csv`

#### `--visualize`
- **Description**: Generate all visualization plots
- **Output**: High-resolution PNG files in `visualizations/output/`
- **Example**: `--visualize`

#### `--log-level <level>`
- **Description**: Set logging verbosity
- **Options**: `DEBUG`, `INFO`, `WARNING`, `ERROR`
- **Default**: `INFO`
- **Example**: `--log-level DEBUG`

### Usage Examples

**Example 1: First-time analysis with visualizations**
```bash
python main.py --benchmark verified --visualize
```

**Example 2: Re-analyze existing data**
```bash
python main.py --benchmark verified --skip-crawl --visualize
```

**Example 3: Include metric correlation analysis**
```bash
python main.py --benchmark verified \
               --metrics-file data/metrics/swe_bench_metrics.csv \
               --visualize
```

**Example 4: Debug mode**
```bash
python main.py --benchmark verified --log-level DEBUG
```

**Example 5: Quick analysis without plots**
```bash
python main.py --benchmark lite --skip-crawl
```

## Expected Data Formats

### Bug Metrics CSV

For metric correlation analysis, provide a CSV file with this structure:

```csv
bug_id,ast_ged,dfg_ged,pdg_ged,loc,cyclomatic_complexity,num_changed_files
bug_0001,45.2,32.1,28.5,120,8,2
bug_0002,89.4,67.3,55.2,245,15,5
bug_0003,12.8,9.4,7.2,45,3,1
```

**Required columns:**
- `bug_id` - Unique identifier matching crawled bug IDs

**Optional metric columns** (include as many as you have):
- `ast_ged` - AST Graph Edit Distance
- `dfg_ged` - Data Flow Graph Edit Distance
- `pdg_ged` - Program Dependence Graph Edit Distance
- `loc` - Lines of Code changed
- `cyclomatic_complexity` - Cyclomatic complexity
- `num_changed_files` - Number of files changed
- Any other numeric metrics

**Notes:**
- The analyzer will auto-detect numeric columns as metrics
- Missing values are handled gracefully
- Metric names are case-insensitive

### Crawled Data Format (Internal)

The crawler saves data in JSON format. You don't need to create these manually:

**Leaderboard data** (`data/raw/leaderboard_*.json`):
```json
[
  {
    "model_id": "gpt4_swe_agent",
    "model_name": "GPT-4 + SWE-agent",
    "score": 38.0,
    "resolved_count": 190
  }
]
```

**Bug results** (`data/raw/bug_results_*.json`):
```json
{
  "gpt4_swe_agent": {
    "bug_0001": true,
    "bug_0002": false
  }
}
```

## Extending to Other Benchmarks

### Step 1: Add Benchmark Configuration

Edit `config/benchmarks.yaml`:

```yaml
benchmarks:
  my_benchmark:
    name: "My Benchmark Name"
    url: "https://mybenchmark.com"
    leaderboard_url: "https://mybenchmark.com/leaderboard.html"
    total_bugs: 500
    api_endpoint: null  # Optional: API endpoint for data
```

### Step 2: Create Custom Crawler

Create `crawlers/my_benchmark_crawler.py`:

```python
from crawlers.base_crawler import BaseCrawler
from typing import Dict, List, Any

class MyBenchmarkCrawler(BaseCrawler):
    """Crawler for My Benchmark."""
    
    def crawl_leaderboard(self) -> List[Dict[str, Any]]:
        """
        Crawl the leaderboard and return model performance data.
        
        Returns:
            List of dicts with keys: model_id, model_name, score, resolved_count
        """
        # Your implementation here
        leaderboard_data = []
        
        # Example: Fetch from API or parse HTML
        # ...
        
        return leaderboard_data
    
    def crawl_bug_results(self, model_id: str) -> Dict[str, bool]:
        """
        Crawl bug-level results for a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dict mapping bug_id to resolved status (bool)
        """
        # Your implementation here
        bug_results = {}
        
        # Example: Fetch detailed results for model
        # ...
        
        return bug_results
```

### Step 3: Register in Main Script

Edit `main.py` to import your crawler:

```python
from crawlers import SWEBenchCrawler, MyBenchmarkCrawler

# Then use it based on benchmark name
if args.benchmark == 'my_benchmark':
    crawler = MyBenchmarkCrawler(selected_benchmark)
else:
    crawler = SWEBenchCrawler(selected_benchmark)
```

### Step 4: Run Your Benchmark

```bash
python main.py --benchmark my_benchmark --visualize
```

## Output Files

### Data Files

All files are timestamped (format: `filename_YYYYMMDD_HHMMSS.csv`):

#### Raw Data (`data/raw/`)
- `leaderboard_*.json` - Crawled leaderboard data
- `bug_results_*.json` - Bug-level results for all models

#### Processed Data (`data/processed/`)
- `bug_data.csv` - Complete bug × model resolution matrix
- `bug_resolution_analysis_*.csv` - Resolution rates and difficulty labels
- `hardest_bugs_*.csv` - Bugs with resolution rate < 25%
- `easiest_bugs_*.csv` - Bugs with resolution rate ≥ 75%
- `tier_statistics_*.csv` - Performance tier statistics
- `tier_similarity_*.csv` - Jaccard similarity between models in same tier
- `common_bugs_by_tier_*.json` - Commonly solved/failed bugs per tier
- `model_agreement_*.csv` - Pairwise model agreement rates
- `unique_solvers_*.csv` - Bugs solved by only one model
- `metric_correlations_*.csv` - Correlation between metrics and difficulty
- `difficulty_category_stats_*.csv` - Metric statistics by difficulty category

### Visualization Files (`visualizations/output/`)

All plots are saved as high-resolution PNG (300 DPI):

- `bug_difficulty_distribution.png` - 2-panel: histogram + category bar chart
- `tier_performance.png` - 4-panel: models/tier, resolution rate, attempts, unique bugs
- `metric_correlations.png` - Horizontal bar chart of top predictive metrics
- `model_agreement_heatmap.png` - Symmetric heatmap of pairwise agreement
- `performance_matrix.png` - Bug × Model heatmap (sampled if > 50 bugs)
- `analysis_summary.txt` - Text report with key statistics

### Log Files (`logs/`)

- `main_*.log` - Detailed execution log with timestamps

## Research Questions Mapping

### RQ1: Which graph representation best predicts LLM repair difficulty?

**Module**: `DifficultyAnalyzer.answer_rq1()`

**Required Input**: Bug metrics CSV with AST/DFG/PDG columns

**Command**:
```bash
python main.py --metrics-file data/metrics/bug_metrics.csv
```

**Output Files**:
- `metric_correlations_*.csv` - Shows correlation for each metric
- Console output - Identifies best predictor

**Interpretation**:
- Look for highest absolute Spearman correlation
- Significant if p-value < 0.05
- Strong correlation: |ρ| > 0.7, Moderate: |ρ| > 0.4

### RQ2: Does semantic complexity outperform syntactic complexity?

**Module**: `DifficultyAnalyzer.answer_rq2()`

**Required Input**: Bug metrics CSV with both semantic (DFG/PDG) and syntactic (AST/LOC) metrics

**Command**: Same as RQ1

**Output Files**: Same as RQ1

**Interpretation**:
- Compares average correlation of semantic vs syntactic metrics
- Reports which category has stronger predictive power

### RQ3: Can we build an accurate difficulty prediction model?

**Modules**: All analyzers provide features

**Approach**:
1. Run complete analysis with metrics
2. Use `metric_correlations_*.csv` to select top features
3. Use `bug_resolution_analysis_*.csv` as target variable (difficulty_label)
4. Train ML model externally (scikit-learn, etc.)

**Top Features** (typically):
- Top 5-10 correlated metrics from correlation analysis
- Model agreement features (from `model_agreement_*.csv`)
- Tier statistics (from `tier_statistics_*.csv`)

## Advanced Usage

### Custom Tier Ranges

Edit `config/benchmarks.yaml`:

```yaml
performance_tiers:
  ranges:
    - [90, 100]
    - [80, 90]
    - [70, 80]
    - [60, 70]
    - [50, 60]
    - [0, 50]
```

### Programmatic Usage

Use modules programmatically in your own scripts:

```python
from analyzers import BugResolverAnalyzer
import pandas as pd

# Load data
analyzer = BugResolverAnalyzer()
analyzer.load_bug_data('data/processed/bug_data.csv')

# Analyze
resolution = analyzer.analyze_bug_resolution_rates()
hardest = analyzer.identify_hardest_bugs(threshold=0.1)

# Use results
print(f"Found {len(hardest)} extremely hard bugs")
```

## Troubleshooting

### "No existing data found"

**Problem**: Running with `--skip-crawl` but no data exists

**Solution**: Run without `--skip-crawl` first to fetch data

### "Metrics file not found"

**Problem**: Specified metrics file doesn't exist

**Solution**: 
- Check file path is correct
- Ensure file is in CSV format
- Verify bug_id column exists

### "No data parsed from HTML"

**Problem**: Crawler couldn't parse leaderboard HTML

**Solution**: 
- The crawler falls back to example data automatically
- Check `data/raw/` for any saved data
- Implement custom parser for your benchmark's HTML structure

### Import Errors

**Problem**: Module import failures

**Solution**:
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check Python version ≥ 3.8
- Run from repository root directory

### Plots Not Generating

**Problem**: No visualizations created despite `--visualize`

**Solution**:
- Check matplotlib backend: may need to install `python3-tk`
- Review logs for specific errors
- Ensure `visualizations/output/` directory exists

## Next Steps for Your Research

### 1. Ground Truth Development
- Manually rate a sample of bugs for difficulty
- Compare manual ratings with metric predictions
- Calculate agreement (Cohen's kappa)

### 2. Feature Engineering
- Extract additional code metrics (e.g., number of methods, class hierarchy depth)
- Create composite metrics (e.g., weighted combination)
- Normalize metrics for fair comparison

### 3. Model Training
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load features and labels
X = correlations[['ast_ged', 'dfg_ged', 'pdg_ged']]
y = resolution_data['difficulty_label']

# Train model
model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=5)
print(f"Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
```

### 4. Cross-Benchmark Validation
- Run analysis on multiple benchmarks
- Compare metric correlations across benchmarks
- Test if findings generalize

### 5. Temporal Analysis
- Track how model performance changes over time
- Analyze if bugs become easier as models improve
- Identify persistent difficult bugs

### 6. Deep Dive Analysis
- Qualitative analysis of hardest bugs
- Categorize bugs by root cause
- Study what makes unique-solver bugs special

## Additional Resources

- **SWE-bench Website**: https://www.swebench.com
- **SWE-bench Paper**: [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://www.swebench.com) (Visit website for latest publications)
- **Issues**: https://github.com/kaileekiki/benchmark-difficulty-analyzer/issues

---

For more information, see the [README.md](../README.md) or open an issue on GitHub.
