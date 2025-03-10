"""
Monte Carlo simulation module for NCAA Basketball Game Simulator
Handles probability distributions and simulation logic
"""

import numpy as np
from scipy import stats
import pandas as pd

class MonteCarloSimulator:
    def __init__(self):
        self.distributions = {}
        self.historical_data = {}
        
    def fit_distributions(self, data, team):
        """
        Fit probability distributions to historical team data
        
        Args:
            data (dict): Historical statistics for the team
            team (str): Team name
        """
        team_distributions = {}
        
        for stat, values in data.items():
            if isinstance(values, (list, np.ndarray)):
                # Fit normal distribution to the data
                params = stats.norm.fit(values)
                team_distributions[stat] = {
                    'distribution': 'normal',
                    'params': params
                }
                
        self.distributions[team] = team_distributions
        self.historical_data[team] = data
        
    def simulate_game(self, team1, team2, n_simulations=1000):
        """
        Run Monte Carlo simulation for a game between two teams
        
        Args:
            team1 (str): First team name
            team2 (str): Second team name
            n_simulations (int): Number of simulations to run
            
        Returns:
            dict: Simulation results including win probabilities and score distributions
        """
        results = {
            'team1_scores': [],
            'team2_scores': [],
            'team1_wins': 0,
            'overtime_games': 0
        }
        
        for _ in range(n_simulations):
            # Simulate game stats for both teams
            team1_stats = self._simulate_team_stats(team1)
            team2_stats = self._simulate_team_stats(team2)
            
            # Calculate scores based on simulated stats
            score1, score2, is_overtime = self._calculate_game_score(team1_stats, team2_stats)
            
            results['team1_scores'].append(score1)
            results['team2_scores'].append(score2)
            
            if score1 > score2:
                results['team1_wins'] += 1
            
            if is_overtime:
                results['overtime_games'] += 1
        
        # Calculate summary statistics
        results['win_probability'] = results['team1_wins'] / n_simulations
        results['overtime_probability'] = results['overtime_games'] / n_simulations
        results['predicted_score'] = (
            np.mean(results['team1_scores']),
            np.mean(results['team2_scores'])
        )
        
        return results
    
    def _simulate_team_stats(self, team):
        """
        Simulate statistics for a single team based on their fitted distributions
        
        Args:
            team (str): Team name
            
        Returns:
            dict: Simulated statistics
        """
        if team not in self.distributions:
            raise ValueError(f"No fitted distributions found for {team}")
            
        simulated_stats = {}
        
        for stat, dist_info in self.distributions[team].items():
            if dist_info['distribution'] == 'normal':
                mean, std = dist_info['params']
                simulated_stats[stat] = np.random.normal(mean, std)
        
        return simulated_stats
    
    def _calculate_game_score(self, team1_stats, team2_stats):
        """
        Calculate game score based on simulated statistics
        
        Args:
            team1_stats (dict): Simulated statistics for team 1
            team2_stats (dict): Simulated statistics for team 2
            
        Returns:
            tuple: (team1_score, team2_score, is_overtime)
        """
        # Base score calculation using offensive and defensive ratings
        base_score1 = team1_stats.get('AdjO', 100) * team2_stats.get('AdjD', 100) / 100
        base_score2 = team2_stats.get('AdjO', 100) * team1_stats.get('AdjD', 100) / 100
        
        # Add random variation
        score1 = int(round(np.random.normal(base_score1, 5)))
        score2 = int(round(np.random.normal(base_score2, 5)))
        
        # Handle overtime
        is_overtime = False
        if abs(score1 - score2) <= 2:
            if np.random.random() < 0.5:  # 50% chance of overtime in very close games
                is_overtime = True
                # Add overtime points
                score1 += np.random.randint(5, 12)
                score2 += np.random.randint(5, 12)
        
        return score1, score2, is_overtime
    
    def get_key_factors(self, team1, team2):
        """
        Analyze key statistical factors that could influence the game
        
        Args:
            team1 (str): First team name
            team2 (str): Second team name
            
        Returns:
            list: Key factors that could influence the game outcome
        """
        factors = []
        
        if team1 not in self.historical_data or team2 not in self.historical_data:
            return factors
        
        team1_data = self.historical_data[team1]
        team2_data = self.historical_data[team2]
        
        # Compare offensive efficiency
        if np.mean(team1_data.get('AdjO', [100])) > np.mean(team2_data.get('AdjO', [100])):
            factors.append(f"{team1} has better offensive efficiency")
        else:
            factors.append(f"{team2} has better offensive efficiency")
        
        # Compare defensive efficiency
        if np.mean(team1_data.get('AdjD', [100])) < np.mean(team2_data.get('AdjD', [100])):
            factors.append(f"{team1} has better defensive efficiency")
        else:
            factors.append(f"{team2} has better defensive efficiency")
        
        # Compare tempo
        team1_tempo = np.mean(team1_data.get('Tempo', [68]))
        team2_tempo = np.mean(team2_data.get('Tempo', [68]))
        if abs(team1_tempo - team2_tempo) > 2:
            faster_team = team1 if team1_tempo > team2_tempo else team2
            factors.append(f"{faster_team} plays at a faster pace")
        
        return factors
    
    def get_confidence_rating(self, simulation_results):
        """
        Calculate confidence rating for simulation results
        
        Args:
            simulation_results (dict): Results from simulate_game
            
        Returns:
            int: Confidence rating from 1-5
        """
        confidence = 3  # Start with medium confidence
        
        # Adjust based on win probability
        win_prob = simulation_results['win_probability']
        if abs(win_prob - 0.5) > 0.2:  # Strong favorite
            confidence += 1
        elif abs(win_prob - 0.5) < 0.05:  # Very even matchup
            confidence -= 1
        
        # Adjust based on score variance
        score_std1 = np.std(simulation_results['team1_scores'])
        score_std2 = np.std(simulation_results['team2_scores'])
        avg_std = (score_std1 + score_std2) / 2
        
        if avg_std < 5:  # Low variance
            confidence += 1
        elif avg_std > 10:  # High variance
            confidence -= 1
        
        # Ensure confidence stays in range 1-5
        return max(1, min(5, confidence))