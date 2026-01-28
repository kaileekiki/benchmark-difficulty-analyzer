"""
SWE-bench specific crawler implementation.
Crawls leaderboard and bug-level results from SWE-bench website.
"""

import requests
import time
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
        self.rate_limit_delay = 1.0  # seconds between requests
    
    def crawl_leaderboard(self) -> List[Dict[str, Any]]:
        """
        Crawl the SWE-bench leaderboard.
        
        Returns:
            List of model performance dictionaries
        """
        self.logger.info(f"Crawling leaderboard from {self.leaderboard_url}")
        
        try:
            # Attempt to fetch the leaderboard
            response = requests.get(self.leaderboard_url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            leaderboard_data = self._parse_leaderboard_html(soup)
            
            if not leaderboard_data:
                self.logger.warning("No data parsed from HTML, falling back to manual input")
                return self._manual_leaderboard_input()
            
            self.logger.info(f"Successfully crawled {len(leaderboard_data)} models")
            return leaderboard_data
            
        except Exception as e:
            self.logger.error(f"Error crawling leaderboard: {e}")
            self.logger.info("Falling back to manual input mode")
            return self._manual_leaderboard_input()
    
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
        except:
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
        Crawl bug-level results for a specific model.
        
        Note: This is a placeholder. Actual implementation would need
        access to detailed bug-level results from the benchmark.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary of bug results
        """
        self.logger.info(f"Attempting to crawl bug results for {model_id}")
        
        # Rate limiting
        time.sleep(self.rate_limit_delay)
        
        # Placeholder: In a real implementation, this would fetch actual bug-level data
        # For now, we'll generate synthetic data based on the model's overall score
        self.logger.warning("Bug-level crawling not implemented, using synthetic data")
        
        return self._generate_synthetic_bug_results(model_id)
    
    def _generate_synthetic_bug_results(self, model_id: str) -> Dict[str, bool]:
        """
        Generate synthetic bug results for demonstration.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary mapping bug IDs to resolution status
        """
        # This is a placeholder - real implementation would fetch actual data
        bug_results = {}
        
        # Generate results for all bugs
        for i in range(1, self.total_bugs + 1):
            bug_id = f"bug_{i:04d}"
            # Simple heuristic: harder bugs have higher IDs
            difficulty = i / self.total_bugs
            # Models are less likely to solve harder bugs
            resolved = difficulty < 0.5  # Placeholder logic
            bug_results[bug_id] = resolved
        
        return bug_results
