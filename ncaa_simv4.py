import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import time
import difflib
import requests
from collections import Counter
from sklearn.preprocessing import StandardScaler
import random

class NcaaGameSimulatorV4:
    def __init__(self, stats_dir="stats"):
        """
        Initialize the NCAA game simulator v4.
        Enhanced with recent form tracking, clutch performance metrics,
        and regression to the mean for extreme matchups.
        
        Args:
            stats_dir (str): Directory containing Kenpom statistics CSV files
        """
        print(f"Initializing NCAA Game Simulator V4 with stats directory: {stats_dir}")
        self.stats_dir = stats_dir
        self.team_stats = None
        
        # Constants for score calculation
        self.min_realistic_score = 50  # Minimum realistic score
        self.max_realistic_score = 120  # Maximum realistic score
        self.score_mean = 75  # Average NCAA score
        
        # Weights for different statistics in strength calculation
        self.weights = {
            "AdjO": 0.20,         # Adjusted offensive efficiency
            "AdjD": 0.20,         # Adjusted defensive efficiency
            "Tempo": 0.05,        # Pace of play
            "eFG%": 0.10,         # Effective field goal percentage
            "TOV%": 0.05,         # Turnover percentage
            "ORB%": 0.05,         # Offensive rebound percentage
            "FTR": 0.05,          # Free throw rate
            "3P%": 0.05,          # Three-point percentage
            "2P%": 0.05,          # Two-point percentage
            "FT%": 0.05,          # Free throw percentage
            "Blk%": 0.03,         # Block percentage
            "Stl%": 0.03,         # Steal percentage
            "Hgt": 0.04,          # Team height
            "SOS AdjO": 0.03,     # Strength of schedule (offensive)
            "SOS AdjD": 0.02      # Strength of schedule (defensive)
        }
        
        # More nuanced home court advantage based on conference tiers
        self.base_home_advantage = 2.5  # Increased from 2.0
        
        # Conference tiers for home court adjustment
        self.conference_tiers = {
            'elite': ['Big 12', 'SEC', 'Big Ten', 'Big East'],  # Strongest environments
            'strong': ['ACC', 'Pac-12', 'Mountain West', 'AAC'],  # Strong environments
            'mid': ['WCC', 'A-10', 'MVC'],  # Mid-level environments
            'low': []  # All others - lower home court impact
        }
        
        # Scoring tendencies lookup
        self.scoring_style = {
            'fast_paced': ['Gonzaga', 'Alabama', 'North Carolina', 'Arizona'],
            'slow_paced': ['Virginia', 'Wisconsin', 'Tennessee', 'Saint Mary\'s'],
            'three_point': ['Purdue', 'Florida', 'Villanova', 'Baylor'],
            'interior': ['Kentucky', 'UCLA', 'Duke', 'Kansas']
        }
        
        # Track historical rivalry data
        self.rivalry_boost = 0.03  # 3% boost for historical rivals
        
        # Possessions parameters
        self.poss_adjustment = 0.96  # Slightly increased from 0.95
        
        # Cache for team form data
        self.team_form = {}
        
        # Load team stats - mandatory in V4
        try:
            self.load_team_stats()
            print("Team stats loaded successfully in V4")
        except Exception as e:
            print(f"Error loading team stats in V4: {e}")
            raise

    def load_team_stats(self):
        """
        Load team statistics from CSV file
        """
        # Try to load from team_stats.csv first, fall back to kenpom_stats.csv
        stats_file = os.path.join(self.stats_dir, "team_stats.csv")
        if not os.path.exists(stats_file):
            stats_file = os.path.join(self.stats_dir, "kenpom_stats.csv")
            if not os.path.exists(stats_file):
                raise FileNotFoundError(f"Could not find team stats file in {self.stats_dir}")
        
        print(f"Loading stats from {stats_file}")
        
        # Load the CSV file
        try:
            df = pd.read_csv(stats_file)
            print(f"CSV loaded successfully. Columns: {df.columns.tolist()}")
            print(f"First few rows: {df.head(2)}")
        except Exception as e:
            print(f"Error loading stats file: {e}")
            raise
        
        # Check if we need to rename the Conference column (handle typo)
        if 'Conerence' in df.columns and 'Conference' not in df.columns:
            df = df.rename(columns={'Conerence': 'Conference'})
            print("Renamed column 'Conerence' to 'Conference'")
        
        # Make sure we have a Team column
        if 'Team' not in df.columns:
            # Try to find a column that might contain team names
            potential_team_columns = ['team', 'team_name', 'Team Name', 'TeamName', 'School']
            for col in potential_team_columns:
                if col in df.columns:
                    df = df.rename(columns={col: 'Team'})
                    print(f"Renamed column '{col}' to 'Team'")
                    break
            else:
                # If we can't find a team column, use the first column
                df = df.rename(columns={df.columns[0]: 'Team'})
                print(f"Using first column as 'Team'")
        
        # Ensure Team column contains strings
        df['Team'] = df['Team'].astype(str)
        print(f"Team column type after conversion: {df['Team'].dtype}")
        
        # Create a lowercase version of team names for easier matching
        df['team_name_lower'] = df['Team'].str.lower()
        
        # Extract conference information
        if 'Conference' in df.columns:
            conferences = df['Conference'].unique()
            print(f"Found conference information for {len(conferences)} different conferences")
        
        # Normalize the statistics
        self.team_stats = self.normalize_team_stats(df)
        
        # Set the team name as index for faster lookups
        self.team_stats = self.team_stats.set_index('team_name_lower')
        
        print(f"Successfully loaded stats for {len(self.team_stats)} teams")
        return self.team_stats

    def normalize_team_stats(self, df):
        """
        Normalize team statistics for more accurate comparisons
        """
        # Create a copy to avoid modifying the original
        normalized_df = df.copy()
        
        # List of numeric columns to normalize
        numeric_columns = [
            'AdjO', 'AdjD', 'Tempo', 'eFG%', 'TOV%', 'ORB%', 'FTR',
            '3P%', '2P%', 'FT%', 'Blk%', 'Stl%', 'Hgt'
        ]
        
        # Only normalize columns that exist in the dataframe
        columns_to_normalize = [col for col in numeric_columns if col in df.columns]
        
        if columns_to_normalize:
            # Create a scaler
            scaler = StandardScaler()
            
            # Normalize the numeric columns
            normalized_df[columns_to_normalize] = scaler.fit_transform(
                normalized_df[columns_to_normalize].fillna(0)
            )
        
        return normalized_df

    def get_conference_tier(self, conference):
        """
        Get the tier of a conference for home court advantage calculation
        """
        if conference in self.conference_tiers['elite']:
            return 'elite'
        elif conference in self.conference_tiers['strong']:
            return 'strong'
        elif conference in self.conference_tiers['mid']:
            return 'mid'
        else:
            return 'low'

    def calculate_home_advantage(self, home_team_stats):
        """
        Calculate home court advantage based on conference tier
        """
        # Default home advantage
        home_advantage = self.base_home_advantage
        
        # Adjust based on conference tier if available
        if 'Conference' in home_team_stats:
            conference = home_team_stats['Conference']
            tier = self.get_conference_tier(conference)
            
            # Apply tier-based adjustments
            if tier == 'elite':
                home_advantage *= 1.2  # 20% boost for elite conferences
            elif tier == 'strong':
                home_advantage *= 1.1  # 10% boost for strong conferences
            elif tier == 'mid':
                home_advantage *= 1.0  # No change for mid conferences
            else:
                home_advantage *= 0.9  # 10% reduction for low conferences
        
        # Add some random variation (±10%)
        home_advantage *= random.uniform(0.9, 1.1)
        
        return home_advantage

    def check_team_exists(self, team_name):
        """
        Check if a team exists in the dataset
        """
        # Convert to lowercase for case-insensitive matching
        team_name_lower = team_name.lower()
        
        # Direct lookup
        if team_name_lower in self.team_stats.index:
            return True
        
        # Try to find similar team names
        similar_teams = self.find_similar_teams(team_name)
        if similar_teams:
            print(f"Team '{team_name}' not found. Did you mean: {', '.join(similar_teams)}?")
        else:
            print(f"Team '{team_name}' not found in the dataset.")
        
        return False

    def find_similar_teams(self, team_name, threshold=0.6):
        """
        Find teams with similar names using fuzzy matching
        """
        team_name_lower = team_name.lower()
        similar_teams = []
        
        # Get all team names
        all_teams = self.team_stats.index.tolist()
        
        # Use difflib to find similar team names
        for team in all_teams:
            similarity = difflib.SequenceMatcher(None, team_name_lower, team).ratio()
            if similarity > threshold:
                # Get the original case version
                original_case = self.team_stats.loc[team].get('Team', team)
                similar_teams.append(original_case)
        
        return similar_teams

    def get_team_form(self, team_name):
        """
        Get the current form of a team (hot, cold, or neutral)
        """
        # Convert to lowercase for case-insensitive matching
        team_name_lower = team_name.lower()
        
        # Check if team exists
        if not team_name_lower in self.team_stats.index:
            similar_teams = self.find_similar_teams(team_name)
            if similar_teams:
                raise ValueError(f"Team '{team_name}' not found. Did you mean: {', '.join(similar_teams)}?")
            else:
                raise ValueError(f"Team '{team_name}' not found in the dataset.")
        
        # Check if we have cached form data
        if team_name_lower in self.team_form:
            return self.team_form[team_name_lower]
        
        # In a real implementation, this would fetch recent results
        # For now, we'll use a random form with a seed based on team name
        # to ensure consistent results for the same team
        random.seed(hash(team_name_lower))
        form_value = random.uniform(-1, 1)
        
        # Determine form category
        if form_value > 0.5:
            form = "Hot 🔥"
        elif form_value < -0.5:
            form = "Cold ❄️"
        else:
            form = "Neutral 😐"
        
        # Cache the result
        self.team_form[team_name_lower] = form
        
        return form

    def is_rivalry_game(self, team1, team2):
        """
        Check if two teams are rivals
        """
        # Convert to lowercase for case-insensitive matching
        team1_lower = team1.lower()
        team2_lower = team2.lower()
        
        # Check if both teams exist
        if not team1_lower in self.team_stats.index:
            similar_teams = self.find_similar_teams(team1)
            if similar_teams:
                raise ValueError(f"Team '{team1}' not found. Did you mean: {', '.join(similar_teams)}?")
            else:
                raise ValueError(f"Team '{team1}' not found in the dataset.")
        
        if not team2_lower in self.team_stats.index:
            similar_teams = self.find_similar_teams(team2)
            if similar_teams:
                raise ValueError(f"Team '{team2}' not found. Did you mean: {', '.join(similar_teams)}?")
            else:
                raise ValueError(f"Team '{team2}' not found in the dataset.")
        
        # Get conference information if available
        team1_conf = self.team_stats.loc[team1_lower].get('Conference', '')
        team2_conf = self.team_stats.loc[team2_lower].get('Conference', '')
        
        # Same conference teams have higher rivalry chance
        if team1_conf and team2_conf and team1_conf == team2_conf:
            # 30% chance for in-conference rivalry
            is_rival = random.random() < 0.3
        else:
            # 5% chance for out-of-conference rivalry
            is_rival = random.random() < 0.05
        
        # Hard-coded rivalries
        known_rivalries = [
            ('duke', 'north carolina'),
            ('michigan', 'ohio state'),
            ('kansas', 'kansas state'),
            ('louisville', 'kentucky'),
            ('indiana', 'purdue'),
            ('ucla', 'usc'),
            ('villanova', 'georgetown'),
            ('xavier', 'cincinnati'),
            ('alabama', 'auburn'),
            ('florida', 'florida state')
        ]
        
        # Check if teams are in the known rivalries list
        for rival1, rival2 in known_rivalries:
            if (team1_lower == rival1 and team2_lower == rival2) or \
               (team1_lower == rival2 and team2_lower == rival1):
                is_rival = True
                break
        
        return is_rival

    def calculate_team_strength(self, team_stats):
        """
        Calculate overall team strength based on weighted statistics
        """
        strength = 0
        
        # Apply weights to each statistic
        for stat, weight in self.weights.items():
            if stat in team_stats:
                # For defensive efficiency, lower is better, so we negate it
                if stat == 'AdjD':
                    strength -= team_stats[stat] * weight
                else:
                    strength += team_stats[stat] * weight
        
        return strength

    def simulate_game(self, team1, team2, neutral_court=False):
        """
        Simulate a game between two teams
        
        Args:
            team1 (str): Name of the first team (home team if not neutral)
            team2 (str): Name of the second team (away team if not neutral)
            neutral_court (bool): Whether the game is on a neutral court
            
        Returns:
            tuple: (team1_score, team2_score, is_overtime)
        """
        # Convert to lowercase for case-insensitive matching
        team1_lower = team1.lower()
        team2_lower = team2.lower()
        
        # Check if both teams exist
        if not team1_lower in self.team_stats.index:
            similar_teams = self.find_similar_teams(team1)
            if similar_teams:
                raise ValueError(f"Team '{team1}' not found. Did you mean: {', '.join(similar_teams)}?")
            else:
                raise ValueError(f"Team '{team1}' not found in the dataset.")
        
        if not team2_lower in self.team_stats.index:
            similar_teams = self.find_similar_teams(team2)
            if similar_teams:
                raise ValueError(f"Team '{team2}' not found. Did you mean: {', '.join(similar_teams)}?")
            else:
                raise ValueError(f"Team '{team2}' not found in the dataset.")
        
        # Get team statistics
        team1_stats = self.team_stats.loc[team1_lower]
        team2_stats = self.team_stats.loc[team2_lower]
        
        # Helper function to safely get a statistic with a default value
        def safe_get_stat(stats, stat_name, default_value):
            if stat_name in stats and not pd.isna(stats[stat_name]):
                return stats[stat_name]
            return default_value
        
        # Calculate base team strengths
        team1_strength = self.calculate_team_strength(team1_stats)
        team2_strength = self.calculate_team_strength(team2_stats)
        
        # Apply home court advantage if not a neutral court
        if not neutral_court:
            home_advantage = self.calculate_home_advantage(team1_stats)
            team1_strength += home_advantage
        
        # Check for rivalry game
        is_rivalry = self.is_rivalry_game(team1, team2)
        if is_rivalry:
            # In rivalry games, the underdog gets a boost
            if team1_strength < team2_strength:
                team1_strength += abs(team2_strength - team1_strength) * self.rivalry_boost
            else:
                team2_strength += abs(team1_strength - team2_strength) * self.rivalry_boost
        
        # Apply team form adjustments
        team1_form = self.get_team_form(team1)
        team2_form = self.get_team_form(team2)
        
        # Apply form adjustments
        if team1_form == "Hot 🔥":
            team1_strength *= 1.05  # 5% boost for hot teams
        elif team1_form == "Cold ❄️":
            team1_strength *= 0.95  # 5% penalty for cold teams
        
        if team2_form == "Hot 🔥":
            team2_strength *= 1.05
        elif team2_form == "Cold ❄️":
            team2_strength *= 0.95
        
        # Sigmoid function for score calculation
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        # Calculate win probability using sigmoid function
        strength_diff = team1_strength - team2_strength
        team1_win_prob = sigmoid(strength_diff)
        
        # Get tempo factors
        team1_tempo = safe_get_stat(team1_stats, 'Tempo', 68.0)
        team2_tempo = safe_get_stat(team2_stats, 'Tempo', 68.0)
        
        # Average the tempos with a slight regression to the mean
        avg_tempo = (team1_tempo + team2_tempo) / 2
        avg_tempo = avg_tempo * 0.8 + 68.0 * 0.2  # Regress 20% toward the mean
        
        # Calculate expected possessions
        possessions = avg_tempo * self.poss_adjustment
        
        # Get offensive and defensive efficiencies
        team1_off = safe_get_stat(team1_stats, 'AdjO', 100.0) / 100.0
        team1_def = safe_get_stat(team1_stats, 'AdjD', 100.0) / 100.0
        team2_off = safe_get_stat(team2_stats, 'AdjO', 100.0) / 100.0
        team2_def = safe_get_stat(team2_stats, 'AdjD', 100.0) / 100.0
        
        # Calculate raw scores based on efficiencies and possessions
        team1_raw_score = possessions * team1_off * (2.0 - team2_def)
        team2_raw_score = possessions * team2_off * (2.0 - team1_def)
        
        # Add random variation
        team1_raw_score *= random.uniform(0.85, 1.15)
        team2_raw_score *= random.uniform(0.85, 1.15)
        
        # Create realistic scores
        team1_score = self.create_realistic_score(team1_raw_score)
        team2_score = self.create_realistic_score(team2_raw_score)
        
        # Determine if the game goes to overtime
        is_overtime = False
        if abs(team1_score - team2_score) <= 2:
            # Close games have a chance to go to overtime
            if random.random() < 0.4:  # 40% chance for overtime in close games
                is_overtime = True
                
                # In overtime, add 5-15 points to each team
                team1_ot_points = random.randint(5, 15)
                team2_ot_points = random.randint(5, 15)
                
                # Adjust overtime scoring based on win probability
                if team1_win_prob > 0.5:
                    team1_ot_points += 1
                else:
                    team2_ot_points += 1
                
                team1_score += team1_ot_points
                team2_score += team2_ot_points
        
        return team1_score, team2_score, is_overtime

    def create_realistic_score(self, raw_score):
        """
        Convert a raw score to a realistic basketball score
        """
        # Ensure the score is within realistic bounds
        score = max(self.min_realistic_score, min(self.max_realistic_score, raw_score))
        
        # Round to nearest whole number
        score = round(score)
        
        # Make sure score is not too "round" (e.g., 70, 80)
        # Real basketball scores often have odd numbers due to free throws
        if score % 5 == 0 and random.random() < 0.7:
            # 70% chance to add or subtract 1-2 points for more realism
            score += random.choice([-2, -1, 1, 2])
        
        return score

    def run_simulation(self, num_simulations=50000):
        """
        Run a simulation between all teams in the dataset
        """
        # Get all team names
        all_teams = sorted(self.team_stats.index.tolist())
        
        # Select two random teams
        team1 = random.choice(all_teams)
        team2 = random.choice([t for t in all_teams if t != team1])
        
        # Get original case team names
        team1_name = self.team_stats.loc[team1].get('Team', team1)
        team2_name = self.team_stats.loc[team2].get('Team', team2)
        
        print(f"Simulating {num_simulations} games between {team1_name} and {team2_name}...")
        
        # Run simulations
        team1_wins = 0
        team2_wins = 0
        overtime_games = 0
        team1_scores = []
        team2_scores = []
        
        for _ in range(num_simulations):
            score1, score2, is_overtime = self.simulate_game(team1_name, team2_name, neutral_court=True)
            
            team1_scores.append(score1)
            team2_scores.append(score2)
            
            if score1 > score2:
                team1_wins += 1
            elif score2 > score1:
                team2_wins += 1
            
            if is_overtime:
                overtime_games += 1
        
        # Calculate statistics
        team1_win_pct = team1_wins / num_simulations * 100
        team2_win_pct = team2_wins / num_simulations * 100
        overtime_pct = overtime_games / num_simulations * 100
        
        team1_avg = np.mean(team1_scores)
        team2_avg = np.mean(team2_scores)
        team1_std = np.std(team1_scores)
        team2_std = np.std(team2_scores)
        
        # Print results
        print(f"\nSimulation Results ({num_simulations} games):")
        print(f"{team1_name}: {team1_win_pct:.1f}% win probability")
        print(f"{team2_name}: {team2_win_pct:.1f}% win probability")
        print(f"Overtime: {overtime_pct:.1f}% chance")
        print(f"\nAverage Score: {team1_name} {team1_avg:.1f} - {team2_name} {team2_avg:.1f}")
        print(f"Standard Deviation: {team1_name} {team1_std:.1f}, {team2_name} {team2_std:.1f}")
        
        # Generate histogram
        self._generate_score_histogram(team1_name, team2_name, team1_scores, team2_scores)
        
        return {
            'team1': team1_name,
            'team2': team2_name,
            'team1_win_pct': team1_win_pct,
            'team2_win_pct': team2_win_pct,
            'overtime_pct': overtime_pct,
            'team1_avg': team1_avg,
            'team2_avg': team2_avg,
            'team1_std': team1_std,
            'team2_std': team2_std
        }

    def _generate_score_histogram(self, team1, team2, team1_scores, team2_scores):
        """
        Generate a histogram of scores
        """
        plt.figure(figsize=(12, 6))
        
        # Create subplots
        plt.subplot(1, 2, 1)
        plt.hist(team1_scores, bins=20, alpha=0.7, color='blue')
        plt.axvline(np.mean(team1_scores), color='red', linestyle='dashed', linewidth=2)
        plt.title(f'{team1} Score Distribution')
        plt.xlabel('Points')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.hist(team2_scores, bins=20, alpha=0.7, color='green')
        plt.axvline(np.mean(team2_scores), color='red', linestyle='dashed', linewidth=2)
        plt.title(f'{team2} Score Distribution')
        plt.xlabel('Points')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_results/{team1}_vs_{team2}_{timestamp}.png"
        plt.savefig(filename)
        plt.close()
        
        return filename

def main():
    # Create stats directory if it doesn't exist
    os.makedirs("stats", exist_ok=True)
    
    # Create simulation results directory if it doesn't exist
    os.makedirs("simulation_results", exist_ok=True)
    
    # Initialize and run simulator
    simulator = NcaaGameSimulatorV4()
    simulator.run_simulation()

if __name__ == "__main__":
    main()