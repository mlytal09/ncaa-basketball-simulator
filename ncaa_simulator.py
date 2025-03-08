import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import time

class NcaaGameSimulator:
    def __init__(self, stats_dir="stats"):
        """
        Initialize the NCAA game simulator.
        
        Args:
            stats_dir (str): Directory containing Kenpom statistics CSV files
        """
        self.stats_dir = stats_dir
        self.team_stats = None
        
        # Define the weights for each statistic based on the provided percentages
        self.weights = {
            # Efficiency Metrics (60%)
            'AdjO': 0.25,
            'AdjD': 0.25,
            'SOS AdjO': 0.05,
            'SOS AdjD': 0.05,
            
            # Four Factors (17%)
            'eFG%': 0.05,
            'TOV%': 0.04,
            'ORB%': 0.04,
            'FTR': 0.04,
            
            # Defensive Stats (14%)
            'Blk%': 0.03,
            'Stl%': 0.03,
            'Def 2P%': 0.03,
            'Def 3P%': 0.03,
            'Def FT%': 0.02,
            
            # Offensive Tendencies (4%)
            'NST%': 0.01,
            'A%': 0.01,
            '3PA%': 0.02,
            
            # Height (1%)
            'Hgt': 0.01,
            
            # Points Distribution (1%)
            '%2P': 0.004,
            '%3P': 0.003,
            '%FT': 0.003,
            
            # Tempo is used directly for possessions calculation
            'Tempo': 0.0
        }
        
        # Home court advantage - reduced from 3.75 to 2.5 points
        self.home_advantage = 2.5
        
        self.load_team_stats()
        
    def load_team_stats(self):
        """Load team statistics from CSV files"""
        try:
            # First try to load from the main directory
            if os.path.exists("kenpom_stats.csv"):
                self.team_stats = pd.read_csv("kenpom_stats.csv")
                print("Loading stats from kenpom_stats.csv in main directory")
            # Then try the stats directory
            elif os.path.exists(f"{self.stats_dir}/kenpom_stats.csv"):
                self.team_stats = pd.read_csv(f"{self.stats_dir}/kenpom_stats.csv")
                print(f"Loading stats from {self.stats_dir}/kenpom_stats.csv")
            else:
                raise FileNotFoundError("No stats file found")
            
            # Convert team names to lowercase for case-insensitive matching
            self.team_stats['team_name_lower'] = self.team_stats['team_name'].str.lower()
            self.team_stats.set_index('team_name', inplace=True)
            
            print(f"Successfully loaded stats for {len(self.team_stats)} teams")
        except FileNotFoundError:
            print(f"Warning: Could not find stats file in main directory or at '{self.stats_dir}/kenpom_stats.csv'")
            print("Using sample data for demonstration. Please add actual stats files.")
            # Create sample data if file not found
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample team stats data for demonstration purposes"""
        sample_teams = ["Gonzaga", "Baylor", "Michigan", "Illinois", "Iowa", 
                        "Ohio State", "Alabama", "Houston", "Arkansas", "Purdue"]
        
        # Generate random stats with realistic ranges
        data = {
            # Efficiency Metrics
            "AdjO": np.random.uniform(100, 120, len(sample_teams)),
            "AdjD": np.random.uniform(85, 100, len(sample_teams)),
            "SOS AdjO": np.random.uniform(105, 110, len(sample_teams)),
            "SOS AdjD": np.random.uniform(100, 105, len(sample_teams)),
            
            # Four Factors
            "eFG%": np.random.uniform(48, 58, len(sample_teams)),
            "TOV%": np.random.uniform(14, 21, len(sample_teams)),
            "ORB%": np.random.uniform(25, 38, len(sample_teams)),
            "FTR": np.random.uniform(25, 40, len(sample_teams)),
            
            # Height
            "Hgt": np.random.uniform(74, 78, len(sample_teams)),
            
            # Points Distribution
            "%2P": np.random.uniform(40, 55, len(sample_teams)),
            "%3P": np.random.uniform(25, 40, len(sample_teams)),
            "%FT": np.random.uniform(18, 22, len(sample_teams)),
            
            # Tempo
            "Tempo": np.random.uniform(65, 75, len(sample_teams)),
            
            # Defensive Stats
            "Blk%": np.random.uniform(8, 13, len(sample_teams)),
            "Stl%": np.random.uniform(7, 11, len(sample_teams)),
            "Def 2P%": np.random.uniform(44, 48, len(sample_teams)),
            "Def 3P%": np.random.uniform(31, 35, len(sample_teams)),
            "Def FT%": np.random.uniform(69, 73, len(sample_teams)),
            
            # Offensive Tendencies
            "NST%": np.random.uniform(7, 12, len(sample_teams)),
            "A%": np.random.uniform(48, 60, len(sample_teams)),
            "3PA%": np.random.uniform(28, 45, len(sample_teams)),
            
            # Home Court Advantage
            "HomeAdvantage": 5.36,
            
            # Lowercase team names for matching
            "team_name_lower": [t.lower() for t in sample_teams]
        }
        
        # Create DataFrame with sample data
        self.team_stats = pd.DataFrame(data, index=sample_teams)
        print("Created sample data for demonstration purposes.")
    
    def check_team_exists(self, team_name):
        """Check if a team exists in the stats database"""
        # First try direct match
        if team_name in self.team_stats.index:
            return team_name
        
        # Try case-insensitive match
        team_lower = team_name.lower()
        matching_teams = self.team_stats[self.team_stats['team_name_lower'] == team_lower]
        
        if not matching_teams.empty:
            # Return the properly capitalized team name
            return matching_teams.index[0]
        
        # No match found, suggest similar teams
        close_matches = self.find_similar_teams(team_name)
        if close_matches:
            suggest = ", ".join(close_matches)
            print(f"Team '{team_name}' not found. Did you mean: {suggest}?")
        else:
            print(f"Team '{team_name}' not found in the database.")
        
        return None
    
    def find_similar_teams(self, team_name, threshold=0.6):
        """Find teams with similar names based on string similarity"""
        from difflib import SequenceMatcher
        
        similar_teams = []
        for team in self.team_stats.index:
            similarity = SequenceMatcher(None, team_name.lower(), team.lower()).ratio()
            if similarity >= threshold:
                similar_teams.append(team)
        
        return similar_teams[:3]  # Return top 3 matches
    
    def calculate_team_strength(self, team_stats):
        """
        Calculate overall team strength based on weighted statistics
        
        Args:
            team_stats: Series containing team statistics
            
        Returns:
            float: Overall team strength score
        """
        strength = 0.0
        
        # Apply weights to each statistic
        for stat, weight in self.weights.items():
            if stat in team_stats and weight > 0:
                # For defensive metrics (lower is better), invert the value
                if stat == 'AdjD' or stat == 'TOV%':
                    # Normalize to a reasonable range
                    if stat == 'AdjD':
                        # Invert AdjD (lower is better) and scale
                        normalized_value = (110 - team_stats[stat]) / 20
                    else:  # TOV%
                        # Invert TOV% (lower is better) and scale
                        normalized_value = (25 - team_stats[stat]) / 10
                    
                    strength += normalized_value * weight
                else:
                    # For offensive metrics (higher is better)
                    # Normalize to a reasonable range based on the statistic
                    if stat == 'AdjO':
                        normalized_value = (team_stats[stat] - 90) / 30
                    elif stat == 'eFG%':
                        normalized_value = team_stats[stat] / 60
                    elif stat == 'ORB%':
                        normalized_value = team_stats[stat] / 40
                    elif stat == 'FTR':
                        normalized_value = team_stats[stat] / 40
                    elif stat == 'Hgt':
                        normalized_value = (team_stats[stat] - 74) / 4
                    elif stat in ['Blk%', 'Stl%']:
                        normalized_value = team_stats[stat] / 15
                    elif stat in ['NST%', '3PA%']:
                        normalized_value = team_stats[stat] / 50
                    elif stat == 'A%':
                        normalized_value = team_stats[stat] / 60
                    elif stat in ['%2P', '%3P', '%FT']:
                        normalized_value = team_stats[stat] / 50
                    else:
                        normalized_value = team_stats[stat] / 100
                    
                    strength += normalized_value * weight
        
        return strength
    
    def simulate_game(self, team1, team2, neutral_court=False):
        """
        Simulate a single game between two teams.
        
        Args:
            team1 (str): Name of the first team (home team if not neutral)
            team2 (str): Name of the second team (away team if not neutral)
            neutral_court (bool): Whether the game is on a neutral court
        
        Returns:
            tuple: (team1_score, team2_score)
        """
        # Get team statistics
        team1_stats = self.team_stats.loc[team1]
        team2_stats = self.team_stats.loc[team2]
        
        # Calculate team strengths using the weighted model
        team1_strength = self.calculate_team_strength(team1_stats)
        team2_strength = self.calculate_team_strength(team2_stats)
        
        # Calculate expected possessions based on teams' tempos
        possessions = (team1_stats["Tempo"] + team2_stats["Tempo"]) / 2
        
        # Calculate base offensive efficiency for each team (points per 100 possessions)
        team1_off = team1_stats["AdjO"]
        team2_off = team2_stats["AdjO"]
        team1_def = team1_stats["AdjD"]
        team2_def = team2_stats["AdjD"]
        
        # Adjust offensive efficiency based on opponent's defense
        team1_adj_off = team1_off * (100 / team2_def)
        team2_adj_off = team2_off * (100 / team1_def)
        
        # Apply the strength differential to the efficiency (greatly reduced impact)
        strength_diff = team1_strength - team2_strength
        # Cap the strength differential to prevent extreme advantages
        strength_diff = max(min(strength_diff, 0.8), -0.8)  # Reduced from 1.0 to 0.8
        team1_adj_off *= (1 + 0.008 * strength_diff)  # Reduced from 0.01 to 0.008
        team2_adj_off *= (1 - 0.008 * strength_diff)  # Reduced from 0.01 to 0.008
        
        # Add home court advantage if not neutral
        if not neutral_court:
            # Apply home court advantage (reduced from 3.75 to 2.5 points)
            # Convert to efficiency adjustment based on expected possessions
            home_eff_boost = self.home_advantage / possessions * 100
            team1_adj_off += home_eff_boost * 0.8  # Reduced impact by 20%
            team2_adj_off -= home_eff_boost * 0.3  # Away team penalty is 30% of the home boost (reduced from 50%)
        
        # Increase standard deviation for more natural score distributions
        poss_stddev = 3.0  # Increased from 2.0 to create more variability
        eff_stddev = 6.0   # Increased from 4.0 to create more variability
        
        # Add a random upset factor (occasionally teams will play much better/worse than expected)
        upset_factor = 1.0
        if np.random.random() < 0.15:  # 15% chance of an upset factor
            # Random factor between 0.85 and 1.15
            upset_factor = np.random.uniform(0.85, 1.15)
            # Apply upset factor to the underdog (team with lower adjusted offensive efficiency)
            if team1_adj_off < team2_adj_off:
                team1_adj_off *= upset_factor
            else:
                team2_adj_off *= upset_factor
        
        # Add additional random noise to create more natural distributions
        # This simulates the natural game-to-game variability in team performance
        team1_performance_factor = np.random.normal(1.0, 0.05)  # 5% standard deviation
        team2_performance_factor = np.random.normal(1.0, 0.05)  # 5% standard deviation
        
        team1_adj_off *= team1_performance_factor
        team2_adj_off *= team2_performance_factor
        
        actual_possessions = np.random.normal(possessions, poss_stddev)
        team1_actual_off = np.random.normal(team1_adj_off, eff_stddev)
        team2_actual_off = np.random.normal(team2_adj_off, eff_stddev)
        
        # Calculate raw scores (points per 100 possessions * actual possessions / 100)
        team1_raw_score = team1_actual_off * actual_possessions / 100
        team2_raw_score = team2_actual_off * actual_possessions / 100
        
        # Apply a more natural score adjustment that makes extreme scores less likely
        # but still possible, centered around the typical college basketball range
        def adjust_score(score):
            # Target mean for college basketball
            target_mean = 72
            
            # Use a gentler adjustment that preserves more of the natural distribution
            if score < 55:
                # Pull very low scores up slightly
                return 55 + (score - 55) * 1.2
            elif score > 90:  # Reduced from 95 to 90
                # Pull very high scores down more aggressively
                return 90 + (score - 90) * 0.4  # Reduced from 0.75 to 0.4
            else:
                # Leave scores in the normal range alone
                return score
        
        # Apply the adjustment and round to integers
        team1_score = max(int(round(adjust_score(team1_raw_score))), 50)
        team2_score = max(int(round(adjust_score(team2_raw_score))), 50)
        
        return team1_score, team2_score
    
    def run_simulation(self, num_simulations=50000):
        """
        Run the game simulation interface.
        
        Args:
            num_simulations (int): Number of simulations to run
        """
        print("\n" + "=" * 50)
        print("NCAA Basketball Game Simulator".center(50))
        print("=" * 50 + "\n")
        
        # Get simulation parameters
        while True:
            neutral_input = input("Is this game on a neutral court? (yes/no): ").strip().lower()
            if neutral_input in ["yes", "y", "true", "1", "no", "n", "false", "0"]:
                neutral_court = neutral_input in ["yes", "y", "true", "1"]
                break
            else:
                print("Please enter 'yes' or 'no'.")
        
        # Use appropriate team labels based on court type
        team1_label = "Team 1" if neutral_court else "Home team"
        team2_label = "Team 2" if neutral_court else "Away team"
        
        # Get first team with case-insensitive matching
        while True:
            team1_input = input(f"Enter the {team1_label}: ").strip()
            team1 = self.check_team_exists(team1_input)
            if team1:
                break
            print("Please try again with a valid team name.")
        
        # Get second team with case-insensitive matching
        while True:
            team2_input = input(f"Enter the {team2_label}: ").strip()
            team2 = self.check_team_exists(team2_input)
            if team2 and team2 != team1:
                break
            elif team2 == team1:
                print("Error: Teams cannot be the same.")
            else:
                print("Please try again with a valid team name.")
        
        print(f"\nSimulating {num_simulations} games between {team1} and {team2}...")
        print(f"Court: {'Neutral' if neutral_court else team1 + ' home'}")
        
        # Run simulations
        start_time = time.time()
        
        team1_wins = 0
        team2_wins = 0
        ties = 0
        team1_scores = []
        team2_scores = []
        
        for _ in range(num_simulations):
            team1_score, team2_score = self.simulate_game(team1, team2, neutral_court)
            team1_scores.append(team1_score)
            team2_scores.append(team2_score)
            
            if team1_score > team2_score:
                team1_wins += 1
            elif team2_score > team1_score:
                team2_wins += 1
            else:
                ties += 1
        
        simulation_time = time.time() - start_time
        
        # Calculate statistics
        avg_team1_score = np.mean(team1_scores)
        avg_team2_score = np.mean(team2_scores)
        std_team1_score = np.std(team1_scores)
        std_team2_score = np.std(team2_scores)
        
        team1_win_pct = team1_wins / num_simulations * 100
        team2_win_pct = team2_wins / num_simulations * 100
        tie_pct = ties / num_simulations * 100
        
        # Find the most common score
        from collections import Counter
        team1_score_counts = Counter(team1_scores)
        team2_score_counts = Counter(team2_scores)
        most_common_team1 = team1_score_counts.most_common(1)[0][0]
        most_common_team2 = team2_score_counts.most_common(1)[0][0]
        
        # Calculate margin of victory
        avg_margin = avg_team1_score - avg_team2_score
        
        # Display results
        print("\n" + "=" * 50)
        print("SIMULATION RESULTS".center(50))
        print("=" * 50)
        print(f"Simulations completed: {num_simulations} ({simulation_time:.2f} seconds)")
        print(f"\n{team1} vs {team2}")
        print(f"Court: {'Neutral' if neutral_court else team1 + ' home'}")
        
        print("\nWin Probability:")
        print(f"{team1}: {team1_win_pct:.1f}%")
        print(f"{team2}: {team2_win_pct:.1f}%")
        print(f"Chance of Overtime: {tie_pct:.1f}%")
        
        print("\nAverage Score:")
        print(f"{team1}: {avg_team1_score:.1f} ± {std_team1_score:.1f}")
        print(f"{team2}: {avg_team2_score:.1f} ± {std_team2_score:.1f}")
        print(f"Margin: {avg_margin:.1f} points")
        
        print("\nMost Common Score:")
        print(f"{team1} {most_common_team1} - {team2} {most_common_team2}")
        
        print("\nPredicted Final:")
        print(f"{team1} {int(round(avg_team1_score))} - {team2} {int(round(avg_team2_score))}")
        
        # Generate and save a histogram of the score distribution
        self._generate_score_histogram(team1, team2, team1_scores, team2_scores)
        
    def _generate_score_histogram(self, team1, team2, team1_scores, team2_scores):
        """Generate and save a histogram of score distributions"""
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot team1 score distribution
            plt.subplot(1, 2, 1)
            plt.hist(team1_scores, bins=30, alpha=0.7, color='blue')
            plt.axvline(np.mean(team1_scores), color='red', linestyle='dashed', linewidth=2)
            plt.title(f"{team1} Score Distribution")
            plt.xlabel("Points")
            plt.ylabel("Frequency")
            
            # Plot team2 score distribution
            plt.subplot(1, 2, 2)
            plt.hist(team2_scores, bins=30, alpha=0.7, color='green')
            plt.axvline(np.mean(team2_scores), color='red', linestyle='dashed', linewidth=2)
            plt.title(f"{team2} Score Distribution")
            plt.xlabel("Points")
            plt.ylabel("Frequency")
            
            # Create output directory if it doesn't exist
            os.makedirs("simulation_results", exist_ok=True)
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_results/{team1}_vs_{team2}_{timestamp}.png"
            plt.tight_layout()
            plt.savefig(filename)
            print(f"\nScore distribution histogram saved to: {filename}")
            
        except Exception as e:
            print(f"Could not generate histogram: {e}")

def main():
    # Create stats directory if it doesn't exist
    os.makedirs("stats", exist_ok=True)
    
    # Initialize and run simulator
    simulator = NcaaGameSimulator()
    simulator.run_simulation()

if __name__ == "__main__":
    main() 