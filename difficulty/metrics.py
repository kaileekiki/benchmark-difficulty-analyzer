"""
Multiple difficulty calculation methods for bug analysis.
Supports various approaches: model success rate, time-based, tier-weighted, and combined.
"""

import re
from typing import Dict, Optional


class DifficultyMetrics:
    """
    Collection of methods to calculate bug difficulty using different approaches.
    """
    
    @staticmethod
    def from_model_success_rate(success_rate: float) -> float:
        """
        Calculate difficulty from model success rate (current method).
        
        Args:
            success_rate: Proportion of models that successfully resolved the bug (0.0 to 1.0)
            
        Returns:
            Difficulty score (0.0 to 1.0, where 1.0 = hardest)
        """
        return 1.0 - success_rate
    
    @staticmethod
    def from_time_difficulty(difficulty_str: str) -> float:
        """
        Calculate difficulty from SWE-bench time-based difficulty field.
        
        Parses strings like '15 min', '30 min', '1 hour', '2 hours' and normalizes to 0-1 scale.
        Assumes max reasonable time is 4 hours (240 minutes).
        
        Args:
            difficulty_str: Time difficulty string (e.g., '15 min', '1 hour')
            
        Returns:
            Normalized difficulty score (0.0 to 1.0, where 1.0 = hardest)
            Returns 0.5 if parsing fails or string is None
        """
        if not difficulty_str or not isinstance(difficulty_str, str):
            return 0.5  # Default to medium difficulty
        
        difficulty_str = difficulty_str.strip().lower()
        
        # Parse the time string
        minutes = 0
        
        # Match patterns like "15 min", "30 minutes", "1 hour", "2 hours"
        hour_match = re.search(r'(\d+\.?\d*)\s*hour', difficulty_str)
        min_match = re.search(r'(\d+\.?\d*)\s*min', difficulty_str)
        
        if hour_match:
            hours = float(hour_match.group(1))
            minutes = hours * 60
        elif min_match:
            minutes = float(min_match.group(1))
        else:
            # Try to extract just a number
            num_match = re.search(r'(\d+\.?\d*)', difficulty_str)
            if num_match:
                # Assume it's minutes if no unit
                minutes = float(num_match.group(1))
            else:
                # Can't parse, return medium difficulty
                return 0.5
        
        # Normalize to 0-1 scale
        # Assume max reasonable time is 4 hours (240 minutes)
        max_minutes = 240.0
        normalized = min(minutes / max_minutes, 1.0)
        
        return normalized
    
    @staticmethod
    def from_model_tiers(tier_results: Dict[str, float], 
                        tier_scores: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate difficulty based on model tier performance.
        
        Higher-tier models (better overall performance) should solve easier bugs.
        If even top-tier models fail, the bug is harder.
        
        Args:
            tier_results: Dictionary mapping tier labels to success rates for this bug
                         e.g., {'90-100': 0.8, '80-90': 0.5, '70-80': 0.3}
            tier_scores: Optional dictionary of tier weights (default: inverse of tier midpoint)
            
        Returns:
            Weighted difficulty score (0.0 to 1.0, where 1.0 = hardest)
        """
        if not tier_results:
            return 0.5  # Default to medium if no data
        
        # Default tier weights: higher tiers get more weight
        if tier_scores is None:
            tier_scores = {}
            for tier_label in tier_results.keys():
                # Parse tier range (e.g., '90-100' -> 95)
                parts = tier_label.split('-')
                if len(parts) == 2:
                    try:
                        tier_min = float(parts[0])
                        tier_max = float(parts[1])
                        tier_midpoint = (tier_min + tier_max) / 2
                        # Weight is proportional to tier quality
                        tier_scores[tier_label] = tier_midpoint / 100.0
                    except ValueError:
                        tier_scores[tier_label] = 0.5
                else:
                    tier_scores[tier_label] = 0.5
        
        # Calculate weighted difficulty
        # difficulty = 1 - weighted_success_rate
        total_weight = 0.0
        weighted_success = 0.0
        
        for tier_label, success_rate in tier_results.items():
            weight = tier_scores.get(tier_label, 0.5)
            weighted_success += success_rate * weight
            total_weight += weight
        
        if total_weight > 0:
            avg_weighted_success = weighted_success / total_weight
            return 1.0 - avg_weighted_success
        else:
            return 0.5
    
    @staticmethod
    def combined(success_rate: float, 
                time_difficulty: float,
                tier_difficulty: Optional[float] = None,
                weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate combined difficulty using weighted average of multiple methods.
        
        Args:
            success_rate: Overall model success rate (0.0 to 1.0)
            time_difficulty: Time-based difficulty score (0.0 to 1.0)
            tier_difficulty: Optional tier-based difficulty score (0.0 to 1.0)
            weights: Dictionary of weights for each component
                    Default: {'success_rate': 0.4, 'time': 0.6}
                    
        Returns:
            Combined difficulty score (0.0 to 1.0, where 1.0 = hardest)
        """
        if weights is None:
            weights = {
                'success_rate': 0.4,
                'time': 0.6
            }
        
        # Calculate individual difficulty scores
        success_difficulty = DifficultyMetrics.from_model_success_rate(success_rate)
        
        # Combine with weights
        combined_score = (
            weights.get('success_rate', 0.4) * success_difficulty +
            weights.get('time', 0.6) * time_difficulty
        )
        
        # If tier difficulty is provided and has weight
        if tier_difficulty is not None and 'tier' in weights:
            # Renormalize weights
            total_weight = weights.get('success_rate', 0.4) + weights.get('time', 0.6) + weights.get('tier', 0.0)
            if total_weight > 0:
                combined_score = (
                    weights.get('success_rate', 0.4) / total_weight * success_difficulty +
                    weights.get('time', 0.6) / total_weight * time_difficulty +
                    weights.get('tier', 0.0) / total_weight * tier_difficulty
                )
        
        # Ensure result is in [0, 1]
        return max(0.0, min(1.0, combined_score))
