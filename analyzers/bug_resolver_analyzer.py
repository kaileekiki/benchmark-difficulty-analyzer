import pandas as pd
import json

class BugResolverAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.bug_data = pd.read_csv(f'{data_dir}/bug_data.csv')

    def analyze_bug_resolution_rates(self):
        analysis = self.bug_data.groupby('bug_id').agg(
            total_models=('model_id', 'count'),
            resolved_count=('resolved', 'sum')
        ).reset_index()
        analysis['resolution_rate'] = analysis['resolved_count'] / analysis['total_models']
        analysis['difficulty_label'] = analysis['resolution_rate'].apply(self._categorize_difficulty)
        return analysis[['bug_id', 'total_models', 'resolved_count', 'resolution_rate', 'difficulty_label']]

    def _categorize_difficulty(self, resolution_rate):
        if resolution_rate >= 0.75:
            return 'Easy'
        elif resolution_rate >= 0.5:
            return 'Medium'
        elif resolution_rate >= 0.25:
            return 'Hard'
        else:
            return 'Very Hard'

    def identify_hardest_bugs(self, threshold):
        analysis = self.analyze_bug_resolution_rates()
        return analysis[analysis['resolution_rate'] < threshold]

    def identify_easiest_bugs(self):
        analysis = self.analyze_bug_resolution_rates()
        return analysis[analysis['difficulty_label'] == 'Easy']

    def analyze_consensus_bugs(self):
        pass_fail = self.bug_data[self.bug_data['resolved'].isin([0, 1])]
        return pass_fail.groupby('bug_id')['resolved'].agg(
            total_responses=('resolved', 'count'),
            consensus=('resolved', 'mean')
        ).reset_index()

    def save_analysis(self, analysis, filename_prefix):
        analysis.to_csv(f'{filename_prefix}_summary.csv', index=False)
        analysis.to_json(f'{filename_prefix}_summary.json', orient='records')
