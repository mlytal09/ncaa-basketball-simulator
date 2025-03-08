import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import time

class NcaaGameSimulatorV2:
    def __init__(self, stats_dir="stats"):
        """
        Initialize the NCAA game simulator v2.
        
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
        
        # Reduced home court advantage to 2.0 points (from previous 2.5)
        self.home_advantage = 2.0
        
        # Typical NCAA scoring ranges - based on historical data
        self.score_mean = 71.5      # Typical average score in college basketball
        self.score_std_dev = 8.5     # Typical standard deviation
        self.min_realistic_score = 50 # Very rare to see scores below 50
        self.max_realistic_score = 90 # Very rare to see scores above 90
        
        # Possessions parameters
        self.poss_adjustment = 0.95  # Adjustment factor for possessions (prevents inflated possessions)
        
        self.load_team_stats()
        
    def load_team_stats(self):
        """Load team statistics from CSV files"""
        try:
            # First try to load from the main directory
            if os.path.exists("kenpom_stats.csv"):
                self.team_stats = pd.read_csv("kenpom_stats.csv")
                print("Loading stats from kenpom_stats.csv in main directory")
            # Then try the stats directory
            elif os.path.exists(os.path.join(self.stats_dir, "kenpom_stats.csv")):
                self.team_stats = pd.read_csv(os.path.join(self.stats_dir, "kenpom_stats.csv"))
                print(f"Loading stats from {self.stats_dir}/kenpom_stats.csv")
            else:
                # If no file is found, use sample data instead of raising an error
                print("No stats file found, using sample data.")
                self._create_sample_data()
                return
            
            # Process the loaded data
            # If 'team_name' is in the DataFrame, set it as the index
            if 'team_name' in self.team_stats.columns:
                # Convert team names to lowercase for case-insensitive matching
                self.team_stats['team_name_lower'] = self.team_stats['team_name'].str.lower()
                self.team_stats.set_index('team_name', inplace=True)
            else:
                # If there's no 'team_name' column, assume the first column is the team name
                first_col = self.team_stats.columns[0]
                self.team_stats.rename(columns={first_col: 'team_name'}, inplace=True)
                self.team_stats['team_name_lower'] = self.team_stats['team_name'].str.lower()
                self.team_stats.set_index('team_name', inplace=True)
            
            print(f"Successfully loaded stats for {len(self.team_stats)} teams")
        except Exception as e:
            print(f"Error loading stats file: {e}")
            print("Using sample data for demonstration.")
            # Create sample data if file not found or there's an error
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
            return matching_teams.index[0]
        
        # If no match, find similar team names
        similar_teams = self.find_similar_teams(team_name)
        if similar_teams:
            suggestion = similar_teams[0]
            print(f"Team '{team_name}' not found. Did you mean: {', '.join(similar_teams)}?")
        else:
            print(f"Team '{team_name}' not found in the database.")
        
        return None
    
    def find_similar_teams(self, team_name, threshold=0.6):
        """Find similar team names based on string similarity"""
        import difflib
        
        team_lower = team_name.lower()
        all_teams = self.team_stats.index.tolist()
        
        # Use difflib to find similar team names
        matches = difflib.get_close_matches(team_lower, 
                                           [t.lower() for t in all_teams], 
                                           n=3, 
                                           cutoff=threshold)
        
        # Map back to actual team names (with proper capitalization)
        similar_teams = []
        for match in matches:
            for team in all_teams:
                if team.lower() == match:
                    similar_teams.append(team)
                    break
        
        return similar_teams
    
    def calculate_team_strength(self, team_stats):
        """
        Calculate overall team strength based on weighted stats.
        
        Args:
            team_stats: Statistics for a single team
            
        Returns:
            float: Team strength score
        """
        strength = 0
        
        # Apply weights to each statistic
        for stat, weight in self.weights.items():
            if stat in team_stats and not pd.isna(team_stats[stat]):
                # Normalize the stat value (higher is better)
                if stat in ["TOV%", "AdjD", "Def 2P%", "Def 3P%", "Def FT%"]:
                    # For these stats, lower is better
                    norm_value = 1 - (team_stats[stat] / 150)  # Normalize to 0-1 range
                else:
                    # For other stats, higher is better
                    norm_value = team_stats[stat] / 150  # Normalize to 0-1 range
                
                strength += weight * norm_value
        
        return strength
    
    def simulate_game(self, team1, team2, neutral_court=False):
        """
        Simulate a single game between two teams with improved score realism.
        
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
        # Apply a scaling factor to prevent inflated possessions
        possessions = (team1_stats["Tempo"] + team2_stats["Tempo"]) / 2 * self.poss_adjustment
        
        # Calculate base offensive efficiency for each team (points per 100 possessions)
        team1_off = team1_stats["AdjO"]
        team2_off = team2_stats["AdjO"]
        team1_def = team1_stats["AdjD"]
        team2_def = team2_stats["AdjD"]
        
        # Adjust offensive efficiency based on opponent's defense
        team1_adj_off = team1_off * (100 / team2_def)
        team2_adj_off = team2_off * (100 / team1_def)
        
        # Apply the strength differential to the efficiency (reduced impact)
        strength_diff = team1_strength - team2_strength
        # Cap the strength differential to prevent extreme advantages
        strength_diff = max(min(strength_diff, 0.6), -0.6)  # Further reduced from 0.8 to 0.6
        team1_adj_off *= (1 + 0.005 * strength_diff)  # Reduced from 0.008 to 0.005
        team2_adj_off *= (1 - 0.005 * strength_diff)  # Reduced from 0.008 to 0.005
        
        # Add home court advantage if not neutral
        if not neutral_court:
            # Convert advantage to efficiency adjustment based on expected possessions
            home_eff_boost = self.home_advantage / possessions * 100
            team1_adj_off += home_eff_boost * 0.7  # Reduced impact by 30%
            team2_adj_off -= home_eff_boost * 0.2  # Away team penalty is 20% of the home boost
        
        # Standard deviations for more natural distributions
        poss_stddev = 2.5  # Slightly reduced for more consistency
        eff_stddev = 5.0   # Balanced for realistic variability
        
        # Add random upset factor (teams occasionally play much better/worse than expected)
        if np.random.random() < 0.15:  # 15% chance of an upset factor
            # Random factor between 0.9 and 1.1 (reduced from 0.85-1.15)
            upset_factor = np.random.uniform(0.9, 1.1)
            # Apply upset factor to the underdog
            if team1_adj_off < team2_adj_off:
                team1_adj_off *= upset_factor
            else:
                team2_adj_off *= upset_factor
        
        # Add game-specific performance variability
        team1_perf = np.random.normal(1.0, 0.04)  # 4% standard deviation (reduced from 5%)
        team2_perf = np.random.normal(1.0, 0.04)  # 4% standard deviation (reduced from 5%)
        
        team1_adj_off *= team1_perf
        team2_adj_off *= team2_perf
        
        # Generate actual possessions and offensive efficiency
        actual_possessions = np.random.normal(possessions, poss_stddev)
        team1_actual_off = np.random.normal(team1_adj_off, eff_stddev)
        team2_actual_off = np.random.normal(team2_adj_off, eff_stddev)
        
        # Calculate raw scores (points per 100 possessions * actual possessions / 100)
        team1_raw_score = team1_actual_off * actual_possessions / 100
        team2_raw_score = team2_actual_off * actual_possessions / 100
        
        # Apply our realistic score distribution model
        team1_score = self.create_realistic_score(team1_raw_score)
        team2_score = self.create_realistic_score(team2_raw_score)
        
        return team1_score, team2_score
    
    def create_realistic_score(self, raw_score):
        """
        Transform a raw score into a realistic college basketball score.
        This uses a combination of statistical methods to create a more
        natural score distribution centered around real college basketball scoring.
        
        Args:
            raw_score (float): The raw calculated score
            
        Returns:
            int: A realistic basketball score
        """
        # Step 1: Initial adjustment based on historical NCAA scoring patterns
        # Apply sigmoid-like transformation to naturally constrain scores
        # to realistic ranges while preserving relative team strength
        
        # Map extreme scores toward a realistic range
        if raw_score < self.score_mean - 20:
            # Very low scores get boosted
            adjusted_score = self.min_realistic_score + (raw_score - (self.score_mean - 20)) * 0.5
        elif raw_score > self.score_mean + 20:
            # Very high scores get dampened
            adjusted_score = self.max_realistic_score - (self.max_realistic_score - (self.score_mean + 20)) * np.exp(-(raw_score - (self.score_mean + 20)) / 15)
        else:
            # Scores in normal range are mostly preserved
            adjusted_score = raw_score
            
        # Step 2: Apply a slight regression toward the mean for all scores
        # This ensures scores cluster around typical basketball scores
        regression_strength = 0.2  # 20% regression toward the mean
        adjusted_score = adjusted_score * (1 - regression_strength) + self.score_mean * regression_strength
        
        # Step 3: Add natural score discretization
        # Basketball scores tend to come in specific patterns (multiples of 2 and 3)
        # This adds a subtle bias toward realistic "looking" basketball scores
        if np.random.random() < 0.7:  # 70% of scores will have this adjustment
            remainder = adjusted_score % 1
            if remainder < 0.3:
                adjusted_score = np.floor(adjusted_score)
            elif remainder > 0.7:
                adjusted_score = np.ceil(adjusted_score)
            else:
                # Slightly favor even numbers (representing more 2-pointers)
                if np.random.random() < 0.6:  # 60% chance to round to even
                    adjusted_score = np.round(adjusted_score / 2) * 2
                else:
                    adjusted_score = np.round(adjusted_score)
        else:
            # Simple rounding for the other 30%
            adjusted_score = np.round(adjusted_score)
        
        # Step 4: Ensure no negative or unrealistically low/high scores
        adjusted_score = max(int(adjusted_score), self.min_realistic_score)
        adjusted_score = min(int(adjusted_score), self.max_realistic_score)
        
        return int(adjusted_score)
    
    def run_simulation(self, num_simulations=50000):
        """
        Run the game simulation interface.
        
        Args:
            num_simulations (int): Number of simulations to run
        """
        print("\n" + "=" * 50)
        print("NCAA Basketball Game Simulator v2".center(50))
        print("=" * 50 + "\n")
        
        # Get simulation parameters
        while True:
            neutral_input = input("Is this game on a neutral court? (yes/no): ").strip().lower()
            if neutral_input in ["yes", "y", "true", "1", "no", "n", "false", "0"]:
                neutral_court = neutral_input in ["yes", "y", "true", "1"]
                break
            else:
                print("Please enter 'yes' or 'no'.")
        
        if neutral_court:
            # Get two teams for neutral court game
            team1 = None
            while team1 is None:
                team1_input = input("Enter the Team 1: ").strip()
                team1 = self.check_team_exists(team1_input)
                if team1 is None:
                    print("Please try again with a valid team name.")
            
            team2 = None
            while team2 is None:
                team2_input = input("Enter the Team 2: ").strip()
                team2 = self.check_team_exists(team2_input)
                if team2 is None:
                    print("Please try again with a valid team name.")
        else:
            # Get home and away teams
            home_team = None
            while home_team is None:
                home_input = input("Enter the Home team: ").strip()
                home_team = self.check_team_exists(home_input)
                if home_team is None:
                    print("Please try again with a valid team name.")
            
            away_team = None
            while away_team is None:
                away_input = input("Enter the Away team: ").strip()
                away_team = self.check_team_exists(away_input)
                if away_team is None:
                    print("Please try again with a valid team name.")
            
            team1 = home_team
            team2 = away_team
        
        print(f"\nSimulating {num_simulations} games between {team1} and {team2}...")
        print(f"Court: {'Neutral' if neutral_court else f'{team1} home'}")
        
        # Run simulations
        start_time = time.time()
        team1_wins = 0
        team2_wins = 0
        ties = 0
        team1_scores = []
        team2_scores = []
        
        for _ in range(num_simulations):
            score1, score2 = self.simulate_game(team1, team2, neutral_court)
            team1_scores.append(score1)
            team2_scores.append(score2)
            
            if score1 > score2:
                team1_wins += 1
            elif score2 > score1:
                team2_wins += 1
            else:
                ties += 1
        
        # Calculate statistics
        team1_avg = np.mean(team1_scores)
        team2_avg = np.mean(team2_scores)
        team1_std = np.std(team1_scores)
        team2_std = np.std(team2_scores)
        margin = team1_avg - team2_avg
        
        # Find most common score
        from collections import Counter
        score_pairs = [(team1_scores[i], team2_scores[i]) for i in range(num_simulations)]
        most_common_score = Counter(score_pairs).most_common(1)[0][0]
        
        # Print results
        print("\n" + "=" * 50)
        print("SIMULATION RESULTS".center(50))
        print("=" * 50)
        
        sim_time = time.time() - start_time
        print(f"\nSimulations completed: {num_simulations} ({sim_time:.2f} seconds)")
        print(f"\n{team1} vs {team2}")
        print(f"Court: {'Neutral' if neutral_court else f'{team1} home'}")
        
        print("\nWin Probability:")
        print(f"{team1}: {team1_wins/num_simulations*100:.1f}%")
        print(f"{team2}: {team2_wins/num_simulations*100:.1f}%")
        print(f"Chance of Overtime: {ties/num_simulations*100:.1f}%")
        
        print("\nAverage Score:")
        print(f"{team1}: {team1_avg:.1f} ± {team1_std:.1f}")
        print(f"{team2}: {team2_avg:.1f} ± {team2_std:.1f}")
        print(f"Margin: {margin:.1f} points")
        
        print("\nMost Common Score:")
        print(f"{team1} {most_common_score[0]} - {team2} {most_common_score[1]}")
        
        print("\nPredicted Final:")
        print(f"{team1} {round(team1_avg)} - {team2} {round(team2_avg)}")
        
        # Generate score distribution histogram
        histogram_path = self._generate_score_histogram(team1, team2, team1_scores, team2_scores)
        print(f"\nScore distribution histogram saved to: {histogram_path}")
    
    def _generate_score_histogram(self, team1, team2, team1_scores, team2_scores):
        """
        Generate a histogram of the score distributions.
        
        Args:
            team1 (str): Name of team 1
            team2 (str): Name of team 2
            team1_scores (list): List of team 1 scores
            team2_scores (list): List of team 2 scores
            
        Returns:
            str: Path to the saved histogram image
        """
        # Create directory if it doesn't exist
        os.makedirs("simulation_results", exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        
        # Team 1 histogram
        plt.subplot(1, 2, 1)
        plt.hist(team1_scores, bins=range(min(team1_scores)-5, max(team1_scores)+5), color='blue', alpha=0.7)
        plt.axvline(np.mean(team1_scores), color='red', linestyle='dashed', linewidth=2)
        plt.xlabel('Points')
        plt.ylabel('Frequency')
        plt.title(f'{team1} Score Distribution')
        
        # Team 2 histogram
        plt.subplot(1, 2, 2)
        plt.hist(team2_scores, bins=range(min(team2_scores)-5, max(team2_scores)+5), color='green', alpha=0.7)
        plt.axvline(np.mean(team2_scores), color='red', linestyle='dashed', linewidth=2)
        plt.xlabel('Points')
        plt.ylabel('Frequency')
        plt.title(f'{team2} Score Distribution')
        
        plt.tight_layout()
        
        # Save histogram
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
    simulator = NcaaGameSimulatorV2()
    simulator.run_simulation()

if __name__ == "__main__":
    main() 