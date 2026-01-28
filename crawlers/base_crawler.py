"""
Base crawler class for benchmark leaderboards.
Provides extensible interface for crawling different benchmarks.
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging


class BaseCrawler(ABC):
    """
    Abstract base class for benchmark crawlers.
    Defines the interface for crawling leaderboard data and bug results.
    """
    
    def __init__(self, benchmark_config: Dict[str, Any], data_dir: str = "data/raw"):
        """
        Initialize the crawler.
        
        Args:
            benchmark_config: Configuration dictionary for the benchmark
            data_dir: Directory to store crawled data
        """
        self.config = benchmark_config
        self.data_dir = data_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
    
    @abstractmethod
    def crawl_leaderboard(self) -> List[Dict[str, Any]]:
        """
        Crawl the leaderboard and return model performance data.
        
        Returns:
            List of dictionaries containing model performance information
        """
        pass
    
    @abstractmethod
    def crawl_bug_results(self, model_id: str) -> Dict[str, Any]:
        """
        Crawl bug-level results for a specific model.
        
        Args:
            model_id: Identifier for the model
            
        Returns:
            Dictionary mapping bug IDs to resolution status
        """
        pass
    
    def save_data(self, data: Any, filename: str) -> str:
        """
        Save data to JSON file with timestamp.
        
        Args:
            data: Data to save
            filename: Base filename (without extension)
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.data_dir, f"{filename}_{timestamp}.json")
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Data saved to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error saving data to {filepath}: {e}")
            raise
    
    def load_latest_data(self, filename_pattern: str) -> Optional[Any]:
        """
        Load the most recent data file matching the pattern.
        
        Args:
            filename_pattern: Pattern to match files (e.g., 'leaderboard')
            
        Returns:
            Loaded data or None if no files found
        """
        try:
            # Find all matching files
            files = [
                f for f in os.listdir(self.data_dir)
                if f.startswith(filename_pattern) and f.endswith('.json')
            ]
            
            if not files:
                self.logger.warning(f"No files found matching pattern: {filename_pattern}")
                return None
            
            # Sort by timestamp (files are named with timestamp)
            latest_file = sorted(files)[-1]
            filepath = os.path.join(self.data_dir, latest_file)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"Loaded data from {filepath}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return None
