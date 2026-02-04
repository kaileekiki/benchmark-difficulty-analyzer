"""
Loader for SWE-bench Verified dataset from HuggingFace.
Loads the 'difficulty' field which represents time-based difficulty.
"""

import logging
import pandas as pd
from typing import Dict, Optional
from datasets import load_dataset


class SWEBenchVerifiedLoader:
    """
    Loads SWE-bench Verified dataset from HuggingFace.
    Extracts time-based difficulty information.
    """
    
    def __init__(self):
        """Initialize the loader."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dataset = None
        self.difficulty_data = None
    
    def load_dataset(self) -> pd.DataFrame:
        """
        Load SWE-bench Verified dataset from HuggingFace.
        
        Returns:
            DataFrame with instance_id and difficulty columns
        """
        self.logger.info("Loading SWE-bench Verified dataset from HuggingFace...")
        
        try:
            # Load the dataset from HuggingFace
            self.dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
            
            # Convert to DataFrame for easier manipulation
            df = self.dataset.to_pandas()
            
            # Extract relevant columns
            # SWE-bench uses 'instance_id' as the bug identifier
            if 'difficulty' in df.columns:
                self.difficulty_data = df[['instance_id', 'difficulty']].copy()
                self.logger.info(f"Loaded difficulty data for {len(self.difficulty_data)} instances")
            else:
                self.logger.warning("'difficulty' field not found in dataset")
                # Create empty dataframe with expected columns
                self.difficulty_data = pd.DataFrame(columns=['instance_id', 'difficulty'])
            
            return self.difficulty_data
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            # Return empty dataframe on error
            return pd.DataFrame(columns=['instance_id', 'difficulty'])
    
    def get_difficulty_for_bug(self, bug_id: str) -> Optional[str]:
        """
        Get difficulty string for a specific bug.
        
        Args:
            bug_id: Bug identifier (instance_id)
            
        Returns:
            Difficulty string (e.g., '15 min', '30 min') or None if not found
        """
        if self.difficulty_data is None:
            self.load_dataset()
        
        if self.difficulty_data is None or len(self.difficulty_data) == 0:
            return None
        
        # Look up the bug
        result = self.difficulty_data[self.difficulty_data['instance_id'] == bug_id]
        
        if len(result) > 0:
            return result.iloc[0]['difficulty']
        else:
            return None
    
    def get_all_difficulties(self) -> Dict[str, str]:
        """
        Get all bug difficulties as a dictionary.
        
        Returns:
            Dictionary mapping bug_id to difficulty string
        """
        if self.difficulty_data is None:
            self.load_dataset()
        
        if self.difficulty_data is None or len(self.difficulty_data) == 0:
            return {}
        
        return dict(zip(
            self.difficulty_data['instance_id'],
            self.difficulty_data['difficulty']
        ))
