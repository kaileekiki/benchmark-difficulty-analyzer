"""
SWE-bench specific crawler implementation.
Crawls leaderboard and bug-level results from SWE-bench website.
"""

import requests
import time
import json
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup
from .base_crawler import BaseCrawler


class SWEBenchCrawler(BaseCrawler):
    """
    Crawler for SWE-bench leaderboard and bug results.
    """
    
    def __init__(self, benchmark_config: Dict[str, Any], data_dir: str = "data/raw"):
        """
        Initialize SWE-bench crawler.
        
        Args:
            benchmark_config: Configuration for SWE-bench
            data_dir: Directory to store crawled data
        """
        super().__init__(benchmark_config, data_dir)
        self.leaderboard_url = benchmark_config.get('leaderboard_url')
        self.total_bugs = benchmark_config.get('total_bugs', 500)
        self.rate_limit_delay = 0.5  # seconds between requests
        # GitHub experiments repository URLs
        self.experiments_repo = "https://api.github.com/repos/swe-bench/experiments"
        self.experiments_raw = "https://raw.githubusercontent.com/swe-bench/experiments/main"
    
    def crawl_leaderboard(self) -> List[Dict[str, Any]]:
        """
        Crawl the SWE-bench leaderboard from GitHub experiments repository.
        
        Returns:
            List of model performance dictionaries
        """
        self.logger.info("Crawling leaderboard from SWE-bench experiments repository")
        
        try:
            # Fetch model directories from GitHub experiments repo
            leaderboard_data = self._fetch_from_github_experiments()
            
            if not leaderboard_data:
                self.logger.warning("No data fetched from GitHub, falling back to manual input")
                return self._manual_leaderboard_input()
            
            # Cache the leaderboard data for bug results extraction
            self._cached_leaderboard = leaderboard_data
            
            self.logger.info(f"Successfully crawled {len(leaderboard_data)} models")
            return leaderboard_data
            
        except Exception as e:
            self.logger.error(f"Error crawling leaderboard: {e}")
            self.logger.info("Falling back to manual input mode")
            return self._manual_leaderboard_input()
    
    def _fetch_from_github_experiments(self) -> List[Dict[str, Any]]:
        """
        Fetch leaderboard data from SWE-bench experiments GitHub repository.
        
        Returns:
            List of model performance dictionaries
        """
        leaderboard_data = []
        
        try:
            # Fetch directory listing of verified evaluation results
            url = f"{self.experiments_repo}/contents/evaluation/verified"
            self.logger.info(f"Fetching model directories from {url}")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            directories = response.json()
            
            # Filter only directories (model results)
            model_dirs = [d for d in directories if d['type'] == 'dir']
            self.logger.info(f"Found {len(model_dirs)} model directories")
            
            for model_dir in model_dirs:
                try:
                    # Rate limiting
                    time.sleep(self.rate_limit_delay)
                    
                    model_name = model_dir['name']
                    self.logger.info(f"  Processing {model_name}...")
                    
                    # Fetch results.json for this model
                    results_url = f"{self.experiments_raw}/evaluation/verified/{model_name}/results/results.json"
                    results_response = requests.get(results_url, timeout=30)
                    
                    if results_response.status_code == 200:
                        results_data = results_response.json()
                        
                        # Extract resolved bugs
                        resolved_bugs = results_data.get('resolved', [])
                        resolved_count = len(resolved_bugs)
                        score = (resolved_count / self.total_bugs) * 100
                        
                        leaderboard_data.append({
                            'model_id': model_name,
                            'model_name': self._format_model_name(model_name),
                            'score': score,
                            'resolved_count': resolved_count,
                            'resolved_bugs': resolved_bugs,
                            'no_generation': results_data.get('no_generation', []),
                            'no_logs': results_data.get('no_logs', [])
                        })
                        
                        self.logger.info(f"    {model_name}: {resolved_count}/{self.total_bugs} ({score:.2f}%)")
                    else:
                        self.logger.warning(f"    Could not fetch results for {model_name}")
                        
                except Exception as e:
                    self.logger.warning(f"    Error processing {model_name}: {e}")
                    continue
            
            # Sort by score descending
            leaderboard_data.sort(key=lambda x: x['score'], reverse=True)
            
            return leaderboard_data
            
        except Exception as e:
            self.logger.error(f"Error fetching from GitHub: {e}")
            return []
    
    def _format_model_name(self, model_id: str) -> str:
        """
        Format model ID into a readable name.
        
        Args:
            model_id: Raw model ID from directory name
            
        Returns:
            Formatted model name
        """
        # Remove date prefix (e.g., "20240728_")
        name = model_id
        parts = name.split('_', 1)
        if len(parts) > 1 and parts[0].isdigit():
            name = parts[1]
        
        # Replace underscores with spaces and capitalize
        name = name.replace('_', ' ').title()
        
        return name
    
    def _parse_leaderboard_html(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Parse leaderboard HTML table.
        
        Args:
            soup: BeautifulSoup object of the page
            
        Returns:
            List of model performance dictionaries
        """
        leaderboard_data = []
        
        # Try to find table with leaderboard data
        # This is a generic parser - actual implementation may need adjustment
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            
            for row in rows[1:]:  # Skip header row
                cols = row.find_all(['td', 'th'])
                
                if len(cols) >= 2:
                    try:
                        model_name = cols[0].get_text(strip=True)
                        # Try to extract percentage or score
                        score_text = cols[1].get_text(strip=True)
                        
                        # Parse score (could be "75.5%" or "75.5" or "377/500")
                        score = self._parse_score(score_text)
                        
                        if model_name and score is not None:
                            leaderboard_data.append({
                                'model_id': model_name.lower().replace(' ', '_'),
                                'model_name': model_name,
                                'score': score,
                                'resolved_count': int(score * self.total_bugs / 100)
                            })
                    except Exception as e:
                        self.logger.debug(f"Error parsing row: {e}")
                        continue
        
        return leaderboard_data
    
    def _parse_score(self, score_text: str) -> Optional[float]:
        """
        Parse score from various formats.
        
        Args:
            score_text: Score text to parse
            
        Returns:
            Score as percentage (0-100) or None
        """
        try:
            # Remove % sign if present
            score_text = score_text.replace('%', '').strip()
            
            # Check if it's a fraction like "377/500"
            if '/' in score_text:
                parts = score_text.split('/')
                numerator = float(parts[0])
                denominator = float(parts[1])
                return (numerator / denominator) * 100
            
            # Otherwise, try to parse as float
            score = float(score_text)
            
            # If score is > 100, it might be absolute count
            if score > 100:
                return (score / self.total_bugs) * 100
            
            return score
        except (ValueError, AttributeError):
            return None
    
    def _manual_leaderboard_input(self) -> List[Dict[str, Any]]:
        """
        Provide fallback manual leaderboard data.
        
        Returns:
            List of example model performance dictionaries
        """
        self.logger.info("Using example leaderboard data")
        
        # Example data based on typical SWE-bench leaderboard
        example_data = [
            {'model_id': 'gpt4_swe_agent', 'model_name': 'GPT-4 + SWE-agent', 'score': 38.0, 'resolved_count': 190},
            {'model_id': 'claude_opus_swe_agent', 'model_name': 'Claude Opus + SWE-agent', 'score': 35.0, 'resolved_count': 175},
            {'model_id': 'gpt4_turbo', 'model_name': 'GPT-4 Turbo', 'score': 28.0, 'resolved_count': 140},
            {'model_id': 'claude_sonnet', 'model_name': 'Claude Sonnet', 'score': 25.0, 'resolved_count': 125},
            {'model_id': 'gemini_pro', 'model_name': 'Gemini Pro', 'score': 20.0, 'resolved_count': 100},
        ]
        
        return example_data
    
    def crawl_bug_results(self, model_id: str) -> Dict[str, Any]:
        """
        Get bug-level results for a specific model from cached leaderboard data.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary of bug results
        """
        self.logger.info(f"Extracting bug results for {model_id}")
        
        # Check if we have cached leaderboard data with detailed results
        if hasattr(self, '_cached_leaderboard'):
            for model in self._cached_leaderboard:
                if model['model_id'] == model_id:
                    return self._create_bug_results_dict(model)
        
        # If not cached, return empty dict - will be populated by leaderboard crawl
        self.logger.warning(f"No cached data for {model_id}, returning empty results")
        return {}
    
    def _create_bug_results_dict(self, model_data: Dict[str, Any]) -> Dict[str, bool]:
        """
        Create bug results dictionary from model data.
        
        Args:
            model_data: Model data from leaderboard
            
        Returns:
            Dictionary mapping bug IDs to resolution status
        """
        bug_results = {}
        
        # Get all bug IDs (format: repo__repo-issue)
        resolved_bugs = set(model_data.get('resolved_bugs', []))
        no_generation = set(model_data.get('no_generation', []))
        no_logs = set(model_data.get('no_logs', []))
        
        # All bugs that were attempted
        all_attempted = resolved_bugs | no_generation | no_logs
        
        # If we have the full list of bugs, we can create complete results
        # For now, just mark the ones we know about
        for bug_id in all_attempted:
            # Format bug_id as bug_XXXX for consistency
            formatted_id = self._format_bug_id(bug_id)
            bug_results[formatted_id] = bug_id in resolved_bugs
        
        # If we don't have all 500 bugs yet, we need to fill in the rest
        # This happens when not all bugs are in the results
        if len(bug_results) < self.total_bugs:
            # Add missing bugs as failed (not in resolved list)
            for i in range(1, self.total_bugs + 1):
                bug_id = f"bug_{i:04d}"
                if bug_id not in bug_results:
                    # We don't have data for this bug, mark as failed
                    bug_results[bug_id] = False
        
        return bug_results
    
    def _format_bug_id(self, raw_bug_id: str) -> str:
        """
        Format raw bug ID to standardized format (bug_XXXX).
        
        Args:
            raw_bug_id: Raw bug ID (e.g., "django__django-11333")
            
        Returns:
            Formatted bug ID (e.g., "bug_0001")
        """
        # For now, we'll use a simple hash-based approach to map bug IDs
        # In a production system, you'd want a proper mapping file
        # We'll use the bug's position in a sorted list of all bug IDs
        
        # Store the mapping for consistency
        if not hasattr(self, '_bug_id_mapping'):
            self._bug_id_mapping = {}
            self._bug_counter = 1
        
        if raw_bug_id not in self._bug_id_mapping:
            self._bug_id_mapping[raw_bug_id] = f"bug_{self._bug_counter:04d}"
            self._bug_counter += 1
        
        return self._bug_id_mapping[raw_bug_id]
    
    def get_bug_id_mapping(self) -> Dict[str, str]:
        """
        Get the mapping of raw bug IDs to formatted bug IDs.
        
        Returns:
            Dictionary mapping raw IDs to formatted IDs
        """
        return getattr(self, '_bug_id_mapping', {})
