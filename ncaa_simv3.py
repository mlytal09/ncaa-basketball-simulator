import pandas as pd
import numpy as np
from scipy import stats
from kenpompy.utils import login
import os
from dotenv import load_dotenv

class NcaaGameSimulatorV3:
    def __init__(self):
        print("Initializing NCAA Game Simulator V3 with KenPom integration...")
        self.team_stats = None
        self.scoring_styles = {}
        self._load_team_stats()
        
    def _load_team_stats(self):
        """Load team statistics from KenPom."""
        try:
            load_dotenv()
            username = os.getenv('KENPOM_USERNAME')
            password = os.getenv('KENPOM_PASSWORD')
            
            if not username or not password:
                raise ValueError("KenPom credentials not found in .env file")
                
            browser = login(username, password)
            # Here you would use kenpompy to get the actual data
            # For now, we'll use sample data
            self._create_sample_data()
            
        except Exception as e:
            print(f"Error loading team stats: {str(e)}")
            self._create_sample_data()
            
        self._update_scoring_styles()
        
    def _create_sample_data(self):
        """Create sample data for testing."""
        sample_data = {
            'Team': ['Kansas', 'Duke', 'North Carolina', 'Kentucky'],
            'Conference': ['Big 12', 'ACC', 'ACC', 'SEC'],
            'AdjOE': [115.2, 114.8, 113.5, 112.9],
            'AdjDE': [90.1, 91.2, 92.5, 91.8],
            'Tempo-Adj': [70.5, 68.2, 71.3, 69.8],
            '3P%': [37.5, 35.2, 34.8, 36.1],
            'eFG%': [54.2, 53.8, 52.9, 53.5],
            'TOV%': [16.5, 17.2, 18.1, 17.5],
            'ORB%': [32.5, 31.8, 30.9, 31.5],
            'FTR': [35.2, 34.8, 33.9, 34.5],
            'HomeAdvantage': [3.5, 3.5, 3.5, 3.5]
        }
        self.team_stats = pd.DataFrame(sample_data)
        
    def _update_scoring_styles(self):
        """Update scoring styles for all teams."""
        if self.team_stats is None:
            return
            
        for _, row in self.team_stats.iterrows():
            team = row['Team']
            tempo = row['Tempo-Adj']
            three_point = row['3P%']
            
            # Classify tempo
            if tempo >= 70:
                style = 'Fast-paced'
            elif tempo <= 65:
                style = 'Methodical'
            else:
                style = 'Balanced'
                
            self.scoring_styles[team] = style
            
    def get_team_scoring_style(self, team):
        """Get the scoring style for a team."""
        return self.scoring_styles.get(team)
        
    def calculate_home_court_advantage(self, team):
        """Calculate home court advantage for a team."""
        base_advantage = self.team_stats.loc[self.team_stats['Team'] == team, 'HomeAdvantage'].iloc[0]
        return base_advantage
        
    def simulate_game(self, team1, team2, n_simulations=1000, neutral_site=False):
        """Simulate a game between two teams."""
        if team1 == team2:
            raise ValueError("Cannot simulate a game between the same team")
            
        try:
            team1_stats = self.team_stats[self.team_stats['Team'] == team1].iloc[0]
            team2_stats = self.team_stats[self.team_stats['Team'] == team2].iloc[0]
        except:
            raise ValueError(f"Could not find statistics for {team1} or {team2}")
            
        # Calculate base offensive and defensive ratings
        team1_off = team1_stats['AdjOE']
        team1_def = team1_stats['AdjDE']
        team2_off = team2_stats['AdjOE']
        team2_def = team2_stats['AdjDE']
        
        # Apply home court advantage if not neutral site
        if not neutral_site:
            home_advantage = self.calculate_home_court_advantage(team1)
            team1_off *= (1 + home_advantage/100)
            team1_def *= (1 - home_advantage/100)
            
        # Run Monte Carlo simulation
        team1_scores = []
        team2_scores = []
        overtime_games = 0
        
        for _ in range(n_simulations):
            # Simulate possessions based on tempo
            possessions = np.random.normal(
                (team1_stats['Tempo-Adj'] + team2_stats['Tempo-Adj'])/2,
                2
            )
            
            # Simulate scores
            team1_score = np.random.normal(team1_off/100 * possessions, 4)
            team2_score = np.random.normal(team2_off/100 * possessions, 4)
            
            # Round scores to integers
            team1_score = round(max(0, team1_score))
            team2_score = round(max(0, team2_score))
            
            # Check for overtime
            if abs(team1_score - team2_score) <= 2:
                if np.random.random() < 0.05:  # 5% chance of overtime
                    overtime_games += 1
                    # Add overtime points (7-10 points per team)
                    team1_score += np.random.randint(7, 11)
                    team2_score += np.random.randint(7, 11)
            
            team1_scores.append(team1_score)
            team2_scores.append(team2_score)
            
        # Calculate results
        team1_wins = sum(t1 > t2 for t1, t2 in zip(team1_scores, team2_scores))
        win_probability = team1_wins / n_simulations
        
        avg_score1 = np.mean(team1_scores)
        avg_score2 = np.mean(team2_scores)
        
        return {
            'win_probability': win_probability,
            'predicted_score': f"{team1}: {avg_score1:.1f} - {team2}: {avg_score2:.1f}",
            'key_factors': self._analyze_key_factors(team1_stats, team2_stats),
            'overtime_probability': overtime_games / n_simulations
        }
        
    def _analyze_key_factors(self, team1_stats, team2_stats):
        """Analyze key factors in the matchup."""
        factors = []
        
        # Offensive efficiency comparison
        if team1_stats['AdjOE'] > team2_stats['AdjOE']:
            factors.append("Better offensive efficiency")
        
        # Defensive efficiency comparison
        if team1_stats['AdjDE'] < team2_stats['AdjDE']:
            factors.append("Better defensive efficiency")
            
        # Tempo advantage
        if abs(team1_stats['Tempo-Adj'] - team2_stats['Tempo-Adj']) > 2:
            if team1_stats['Tempo-Adj'] > team2_stats['Tempo-Adj']:
                factors.append("Faster pace preference")
            else:
                factors.append("Slower pace preference")
                
        return factors