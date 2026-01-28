# Benchmark Difficulty Analyzer

A comprehensive research tool for analyzing bug difficulty in automated program repair (APR) benchmarks. This project crawls leaderboard data, analyzes bug resolution rates, and correlates bug metrics with model performance to understand what makes bugs difficult for LLMs to repair.

## 🎯 Project Overview

This analyzer helps researchers understand:
- **What makes bugs difficult?** - Correlates code complexity metrics with resolution rates
- **Which metrics predict difficulty?** - Tests AST, DFG, PDG complexity metrics
- **How do models compare?** - Analyzes agreement and unique strengths between models

### Research Context

**Goal**: Measure bug difficulty using Graph Edit Distance (GED) on various code representations (AST, DFG, PDG)

**Research Questions**:
- **RQ1**: Which graph representation (AST/DFG/PDG) best predicts LLM repair difficulty?
- **RQ2**: Does semantic complexity (DFG/PDG) outperform syntactic complexity (AST/LOC)?
- **RQ3**: Can we build an accurate difficulty prediction model?

## ✨ Features

- **Leaderboard Crawling**: Automatically fetch model performance from benchmark websites
- **Bug-Level Analysis**: Calculate resolution rates and difficulty categories for each bug
- **Performance Tiers**: Group models into performance ranges and analyze patterns
- **Metric Correlation**: Correlate code metrics (AST-GED, DFG-GED, PDG-GED, LOC) with difficulty
- **Model Comparison**: Calculate pairwise agreement and identify model clusters
- **Publication-Quality Visualizations**: Generate high-resolution plots for research papers
- **Extensible Design**: Easily add support for new benchmarks

## 📦 Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kaileekiki/benchmark-difficulty-analyzer.git
cd benchmark-difficulty-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

Analyze SWE-bench Verified with visualizations:
```bash
python main.py --benchmark verified --visualize
```

### Skip Crawling (Use Existing Data)

```bash
python main.py --benchmark verified --skip-crawl --visualize
```

### Include Metric Correlation Analysis

If you have bug metrics (AST-GED, DFG-GED, etc.):
```bash
python main.py --benchmark verified --metrics-file data/metrics/bug_metrics.csv --visualize
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--benchmark` | Benchmark to analyze (verified, lite) | `swe_bench_verified` |
| `--skip-crawl` | Skip crawling and use existing data | `False` |
| `--metrics-file` | Path to bug metrics CSV file | `None` |
| `--visualize` | Generate visualization plots | `False` |
| `--log-level` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |

## 📊 Project Structure

```
benchmark-difficulty-analyzer/
├── main.py                      # Main orchestration script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
│
├── config/                      # Configuration files
│   ├── benchmarks.yaml          # Benchmark configurations
│   └── models.yaml              # Model tracking
│
├── crawlers/                    # Leaderboard crawlers
│   ├── __init__.py
│   ├── base_crawler.py          # Abstract base crawler
│   └── swe_bench_crawler.py     # SWE-bench implementation
│
├── analyzers/                   # Analysis modules
│   ├── __init__.py
│   ├── bug_resolver_analyzer.py # Bug resolution analysis
│   ├── tier_analyzer.py         # Performance tier analysis
│   ├── difficulty_analyzer.py   # Metric correlation analysis
│   └── model_comparison.py      # Model comparison analysis
│
├── visualizations/              # Visualization modules
│   ├── __init__.py
│   ├── plotters.py              # Plotting functions
│   └── output/                  # Generated plots
│
├── utils/                       # Utility modules
│   ├── __init__.py
│   └── logger.py                # Logging configuration
│
├── data/                        # Data storage
│   ├── raw/                     # Raw crawled data
│   ├── processed/               # Processed analysis results
│   └── metrics/                 # Bug metrics (user-provided)
│
├── logs/                        # Log files
│
└── docs/                        # Documentation
    └── USAGE.md                 # Detailed usage guide
```

## 📈 Analysis Workflow

1. **Crawl Leaderboard** → Fetch model performance data from benchmark website
2. **Bug Resolution Analysis** → Calculate resolution rates and difficulty categories
3. **Tier Analysis** → Group models by performance and analyze patterns
4. **Difficulty Correlation** → Correlate metrics with resolution difficulty (if metrics provided)
5. **Model Comparison** → Calculate agreement and identify clusters
6. **Visualization** → Generate publication-quality plots
7. **Summary Report** → Print and save key statistics

## 📂 Output Files

### Data Files (CSV & JSON)

All analysis results are saved in both CSV and JSON formats with timestamps:

- `data/processed/bug_resolution_analysis_*.csv` - Bug-level resolution rates
- `data/processed/hardest_bugs_*.csv` - Very hard bugs (resolution rate < 25%)
- `data/processed/easiest_bugs_*.csv` - Easy bugs (resolution rate ≥ 75%)
- `data/processed/tier_statistics_*.csv` - Performance tier statistics
- `data/processed/tier_similarity_*.csv` - Jaccard similarity within tiers
- `data/processed/model_agreement_*.csv` - Pairwise model agreement
- `data/processed/metric_correlations_*.csv` - Correlation results (if metrics provided)

### Visualizations (PNG)

High-resolution (300 DPI) plots saved to `visualizations/output/`:

- `bug_difficulty_distribution.png` - Histogram and category breakdown
- `tier_performance.png` - 4-panel tier analysis
- `metric_correlations.png` - Top predictive metrics (if metrics provided)
- `model_agreement_heatmap.png` - Pairwise model agreement
- `performance_matrix.png` - Bug × Model resolution matrix
- `analysis_summary.txt` - Text summary report

## 🔧 Extending to Other Benchmarks

### 1. Add Benchmark Configuration

Edit `config/benchmarks.yaml`:

```yaml
benchmarks:
  your_benchmark:
    name: "Your Benchmark Name"
    url: "https://your-benchmark.com"
    leaderboard_url: "https://your-benchmark.com/leaderboard.html"
    total_bugs: 1000
    api_endpoint: null  # Optional API endpoint
```

### 2. Create Custom Crawler

Create `crawlers/your_benchmark_crawler.py`:

```python
from crawlers.base_crawler import BaseCrawler

class YourBenchmarkCrawler(BaseCrawler):
    def crawl_leaderboard(self):
        # Implement leaderboard crawling
        pass
    
    def crawl_bug_results(self, model_id):
        # Implement bug-level result crawling
        pass
```

### 3. Use Your Benchmark

```bash
python main.py --benchmark your_benchmark --visualize
```

## 📚 Research Questions Mapping

| Research Question | Analysis Module | Output |
|-------------------|-----------------|--------|
| **RQ1**: Which graph representation best predicts difficulty? | `DifficultyAnalyzer.answer_rq1()` | `metric_correlations.csv` |
| **RQ2**: Does semantic outperform syntactic complexity? | `DifficultyAnalyzer.answer_rq2()` | `metric_correlations.csv` |
| **RQ3**: Can we build a difficulty prediction model? | All analyzers | Feature importance from correlations |

## 📋 Expected Data Formats

### Bug Metrics CSV

If providing your own bug metrics, use this format:

```csv
bug_id,ast_ged,dfg_ged,pdg_ged,loc,cyclomatic_complexity
bug_0001,45.2,32.1,28.5,120,8
bug_0002,89.4,67.3,55.2,245,15
...
```

Required columns:
- `bug_id` - Unique bug identifier
- At least one metric column (e.g., `ast_ged`, `dfg_ged`, `pdg_ged`, `loc`)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

## 🙏 Citation

If you use this tool in your research, please cite:

```bibtex
@software{benchmark_difficulty_analyzer,
  title = {Benchmark Difficulty Analyzer},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/kaileekiki/benchmark-difficulty-analyzer}
}
```

## 🔍 Next Steps for Your Research

1. **Ground Truth Development**: Collect manual difficulty ratings to validate metrics
2. **Feature Engineering**: Identify additional code complexity features
3. **Model Training**: Train ML models to predict difficulty using top features
4. **Cross-Benchmark Validation**: Test findings across multiple benchmarks
5. **Temporal Analysis**: Analyze how model performance evolves over time

---

**Built with ❤️ for the automated program repair research community**
