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

class NcaaGameSimulatorV3:
    def __init__(self, stats_dir="stats"):
        """
        Initialize the NCAA game simulator v3.
        Enhanced with recent form tracking, clutch performance metrics,
        and regression to the mean for extreme matchups.
        
        Args:
            stats_dir (str): Directory containing Kenpom statistics CSV files
        """
        self.stats_dir = stats_dir
        self.team_stats = None
        
        # Refined weights based on predictive analysis
        self.weights = {
            # Efficiency Metrics (65%)
            'AdjO': 0.30,  # Increased from 0.25
            'AdjD': 0.30,  # Increased from 0.25
            'SOS AdjO': 0.03,  # Decreased from 0.05
            'SOS AdjD': 0.02,  # Decreased from 0.05
            
            # Four Factors (18%)
            'eFG%': 0.06,  # Increased from 0.05
            'TOV%': 0.04,
            'ORB%': 0.05,  # Increased from 0.04
            'FTR': 0.03,  # Decreased from 0.04
            
            # Defensive Stats (12%)
            'Blk%': 0.02,  # Decreased from 0.03
            'Stl%': 0.03,
            'Def 2P%': 0.03,
            'Def 3P%': 0.03,
            'Def FT%': 0.01,  # Decreased from 0.02
            
            # Offensive Tendencies (4%)
            'NST%': 0.01,
            'A%': 0.01,
            '3PA%': 0.02,
            
            # Height (1%)
            'Hgt': 0.01,
            
            # Points Distribution (unchanged)
            '%2P': 0.004,
            '%3P': 0.003,
            '%FT': 0.003,
            
            # Tempo is used directly for possessions calculation
            'Tempo': 0.0
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
        
        # Typical scoring ranges - slightly refined
        self.score_mean = 72.0  # Updated from 71.5
        self.score_std_dev = 8.0  # Updated from 8.5
        self.min_realistic_score = 55  # Increased from 50
        self.max_realistic_score = 95  # Increased from 90
        
        # Upset parameters
        self.base_upset_chance = 0.15  # Baseline chance for upset factors
        self.min_upset_chance = 0.03  # Minimum upset chance even for big mismatches
        
        # Possessions parameters
        self.poss_adjustment = 0.96  # Slightly increased from 0.95
        
        # Cache for team form data
        self.team_form = {}
        
        # Load team stats - mandatory in V3
        self.load_team_stats()
        
    def load_team_stats(self):
        """Load team statistics from CSV files - now required in V3"""
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
                raise FileNotFoundError("No KenPom stats file found. Please ensure kenpom_stats.csv exists.")
            
            # Process the loaded data
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
            
            # Add conference tier information if available
            if 'Conference' in self.team_stats.columns:
                self.team_stats['conf_tier'] = self.team_stats['Conference'].apply(self.get_conference_tier)
            
            # Normalize stats for better comparison
            # This helps ensure all stats contribute appropriately
            self.normalize_team_stats()
            
            print(f"Successfully loaded stats for {len(self.team_stats)} teams")
        except Exception as e:
            print(f"Error loading stats file: {e}")
            raise RuntimeError("Cannot proceed without valid KenPom statistics file")
    
    def normalize_team_stats(self):
        """
        Normalize team stats using feature scaling to improve comparison.
        Only applied to numeric columns that aren't already on a fixed scale.
        """
        # Select columns that should be normalized (numeric columns only)
        numeric_cols = self.team_stats.select_dtypes(include=np.number).columns.tolist()
        
        # Remove columns that shouldn't be normalized
        skip_cols = ['team_name_lower', 'Rk', 'W-L']
        normalize_cols = [col for col in numeric_cols if col not in skip_cols]
        
        if normalize_cols:
            # Create normalized versions of important stats
            for col in normalize_cols:
                if col in self.team_stats.columns:
                    # Calculate mean and std for normalization
                    mean = self.team_stats[col].mean()
                    std = self.team_stats[col].std()
                    if std > 0:  # Avoid division by zero
                        # Create normalized column (z-score)
                        self.team_stats[f"{col}_norm"] = (self.team_stats[col] - mean) / std
    
    def get_conference_tier(self, conference):
        """Determine the tier of a conference for home court advantage"""
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
        Calculate home court advantage based on team's conference and other factors.
        
        Args:
            home_team_stats: Statistics for the home team
            
        Returns:
            float: Home court advantage in points
        """
        base_advantage = self.base_home_advantage
        
        # Adjust based on conference tier if available
        if 'Conference' in home_team_stats:
            conference = home_team_stats['Conference']
            if conference in self.conference_tiers['elite']:
                base_advantage *= 1.2  # 20% boost
            elif conference in self.conference_tiers['strong']:
                base_advantage *= 1.1  # 10% boost
            elif conference in self.conference_tiers['mid']:
                base_advantage *= 1.0  # No change
            else:
                base_advantage *= 0.9  # 10% reduction
        
        # Boost for teams that typically perform better at home
        # Could be calculated from home/away split data if available
        if 'home_performance' in home_team_stats:
            base_advantage *= (1 + home_team_stats['home_performance'] * 0.1)
        
        return base_advantage
    
    def check_team_exists(self, team_name):
        """Check if a team exists in the stats database with improved name matching"""
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
    
    def get_team_form(self, team_name):
        """
        Try to retrieve recent form data for a team.
        In a real implementation, this would connect to a live API.
        For this version, we'll simulate it.
        
        Args:
            team_name: Name of the team
            
        Returns:
            dict: Form data including recent wins/losses and scoring trends
        """
        # Check if we've already retrieved form for this team
        if team_name in self.team_form:
            return self.team_form[team_name]
        
        # For a real implementation, you would get this data from an API
        # Here, we'll generate some simulated form data based on team strength
        if team_name in self.team_stats.index:
            team_stats = self.team_stats.loc[team_name]
            team_strength = self.calculate_team_strength(team_stats)
            
            # Simulate win percentage based on team strength
            win_pct = 0.3 + team_strength * 0.5  # Between 30% and 80%
            
            # Simulate recent form - slightly random but weighted by team strength
            recent_form = np.random.normal(win_pct, 0.15)
            recent_form = max(0.1, min(0.9, recent_form))  # Keep between 10% and 90%
            
            # Store form data
            form_data = {
                'recent_win_pct': recent_form,
                'momentum': np.random.normal(0, 0.05),  # Random momentum factor
                'last_updated': datetime.now()
            }
            
            # Cache the result
            self.team_form[team_name] = form_data
            return form_data
        
        # Default form data if team not found
        return {'recent_win_pct': 0.5, 'momentum': 0, 'last_updated': datetime.now()}
    
    def is_rivalry_game(self, team1, team2):
        """
        Check if two teams have a historical rivalry.
        This would ideally use historical data.
        
        Args:
            team1, team2: Names of the teams
            
        Returns:
            bool: Whether the teams are rivals
        """
        # List of known rivalries (could be expanded or loaded from data)
        rivalries = [
            ('Duke', 'North Carolina'),
            ('Kentucky', 'Louisville'),
            ('Kansas', 'Kansas State'),
            ('Indiana', 'Purdue'),
            ('Michigan', 'Michigan State'),
            ('UCLA', 'USC'),
            ('Alabama', 'Auburn'),
            ('Villanova', 'Georgetown'),
            ('Xavier', 'Cincinnati')
            # Add more rivalries as needed
        ]
        
        # Check if teams are in the rivalry list (in either order)
        for rival1, rival2 in rivalries:
            if (team1 == rival1 and team2 == rival2) or (team1 == rival2 and team2 == rival1):
                return True
        
        return False
    
    def calculate_team_strength(self, team_stats):
        """
        Calculate overall team strength based on weighted stats.
        Enhanced with better normalization and handling of missing values.
        
        Args:
            team_stats: Statistics for a single team
            
        Returns:
            float: Team strength score
        """
        strength = 0
        
        # Apply weights to each statistic - prioritizing normalized versions if available
        for stat, weight in self.weights.items():
            # Try to use normalized version first
            norm_stat = f"{stat}_norm"
            if norm_stat in team_stats and not pd.isna(team_stats[norm_stat]):
                # For normalized stats, higher is always better
                norm_value = (team_stats[norm_stat] + 2) / 4  # Scale from roughly -2..2 to 0..1
                # Cap between 0 and 1
                norm_value = max(0, min(1, norm_value))
                strength += weight * norm_value
            elif stat in team_stats and not pd.isna(team_stats[stat]):
                # Fall back to non-normalized stats with manual normalization
                if stat in ["TOV%", "AdjD", "Def 2P%", "Def 3P%", "Def FT%"]:
                    # For these stats, lower is better
                    if stat == "TOV%":
                        norm_value = 1 - ((team_stats[stat] - 8) / (25 - 8))
                    elif stat == "AdjD":
                        norm_value = 1 - ((team_stats[stat] - 85) / (110 - 85))
                    elif stat == "Def 2P%":
                        norm_value = 1 - ((team_stats[stat] - 40) / (55 - 40))
                    elif stat == "Def 3P%":
                        norm_value = 1 - ((team_stats[stat] - 28) / (38 - 28))
                    elif stat == "Def FT%":
                        norm_value = 1 - ((team_stats[stat] - 65) / (80 - 65))
                else:
                    # For other stats, higher is better
                    if stat == "AdjO":
                        norm_value = (team_stats[stat] - 95) / (120 - 95)
                    elif stat == "eFG%":
                        norm_value = (team_stats[stat] - 45) / (60 - 45)
                    elif stat == "ORB%":
                        norm_value = (team_stats[stat] - 20) / (40 - 20)
                    elif stat == "FTR":
                        norm_value = (team_stats[stat] - 25) / (45 - 25)
                    elif stat == "Hgt":
                        norm_value = (team_stats[stat] - 74) / (79 - 74)
                    else:
                        # General normalization
                        norm_value = team_stats[stat] / 100
                
                # Ensure values are between 0 and 1
                norm_value = max(0, min(1, norm_value))
                strength += weight * norm_value
        
        return strength
    
    def simulate_game(self, team1, team2, neutral_court=False):
        """
        Simulate a single game between two teams with improved realism.
        Enhanced with team form considerations and rivalry adjustments.
        
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
        
        # Get recent form data
        team1_form = self.get_team_form(team1)
        team2_form = self.get_team_form(team2)
        
        # Calculate team strengths using the weighted model
        team1_strength = self.calculate_team_strength(team1_stats)
        team2_strength = self.calculate_team_strength(team2_stats)
        
        # Apply form adjustments
        team1_form_factor = 1.0 + (team1_form['recent_win_pct'] - 0.5) * 0.1 + team1_form['momentum']
        team2_form_factor = 1.0 + (team2_form['recent_win_pct'] - 0.5) * 0.1 + team2_form['momentum']
        
        team1_strength *= team1_form_factor
        team2_strength *= team2_form_factor
        
        # Check for rivalry game
        if self.is_rivalry_game(team1, team2):
            # In rivalry games, teams tend to play closer to their potential
            # and the weaker team often plays better than expected
            if team1_strength < team2_strength:
                team1_strength += self.rivalry_boost
            else:
                team2_strength += self.rivalry_boost
        
        # Calculate expected possessions based on teams' tempos
        possessions = (team1_stats["Tempo"] + team2_stats["Tempo"]) / 2 * self.poss_adjustment
        
        # Calculate base offensive efficiency for each team (points per 100 possessions)
        team1_off = team1_stats["AdjO"]
        team2_off = team2_stats["AdjO"]
        team1_def = team1_stats["AdjD"]
        team2_def = team2_stats["AdjD"]
        
        # Adjust offensive efficiency based on opponent's defense
        team1_adj_off = team1_off * (100 / team2_def)
        team2_adj_off = team2_off * (100 / team1_def)
        
        # Apply the strength differential to the efficiency with regression to the mean
        strength_diff = team1_strength - team2_strength
        
        # Regression to the mean - stronger for larger differentials
        # This ensures more realistic predictions for mismatched teams
        regression_factor = min(0.5, abs(strength_diff) * 3)  # Up to 50% regression
        
        # Apply regression to strength differential
        reg_strength_diff = strength_diff * (1 - regression_factor)
        
        # Cap the strength differential to prevent extreme advantages
        reg_strength_diff = max(min(reg_strength_diff, 0.5), -0.5)
        
        # Apply adjusted differential to efficiencies
        team1_adj_off *= (1 + 0.004 * reg_strength_diff)  # Reduced impact
        team2_adj_off *= (1 - 0.004 * reg_strength_diff)  # Reduced impact
        
        # Add home court advantage if not neutral
        if not neutral_court:
            # Calculate home court advantage based on team factors
            home_advantage = self.calculate_home_advantage(team1_stats)
            
            # Convert advantage to efficiency adjustment
            home_eff_boost = home_advantage / possessions * 100
            team1_adj_off += home_eff_boost * 0.7
            team2_adj_off -= home_eff_boost * 0.2
        
        # Standard deviations for more natural distributions
        poss_stddev = 2.5
        eff_stddev = 5.0
        
        # Calculate upset chance that scales with strength differential
        # Even extreme mismatches should have some chance of upset
        upset_chance = max(
            self.min_upset_chance,
            self.base_upset_chance * (1 - abs(reg_strength_diff))
        )
        
        # Apply random upset factor
        if np.random.random() < upset_chance:
            # Upset factors have more impact in mismatched games
            upset_magnitude = 0.1 + 0.1 * abs(reg_strength_diff)  # 10-20% swing
            
            # Apply to the underdog
            if team1_adj_off < team2_adj_off:
                team1_adj_off *= (1 + upset_magnitude)
            else:
                team2_adj_off *= (1 + upset_magnitude)
        
        # Add game-specific performance variability
        team1_perf = np.random.normal(1.0, 0.04)
        team2_perf = np.random.normal(1.0, 0.04)
        
        team1_adj_off *= team1_perf
        team2_adj_off *= team2_perf
        
        # Generate actual possessions and offensive efficiency
        actual_possessions = np.random.normal(possessions, poss_stddev)
        team1_actual_off = np.random.normal(team1_adj_off, eff_stddev)
        team2_actual_off = np.random.normal(team2_adj_off, eff_stddev)
        
        # Calculate raw scores
        team1_raw_score = team1_actual_off * actual_possessions / 100
        team2_raw_score = team2_actual_off * actual_possessions / 100
        
        # Apply our realistic score distribution model
        team1_score = self.create_realistic_score(team1_raw_score)
        team2_score = self.create_realistic_score(team2_raw_score)
        
        # Resolve tied games - slightly favor the team with higher pre-game strength
        if team1_score == team2_score:
            if np.random.random() < (0.5 + reg_strength_diff * 0.3):
                team1_score += 1
            else:
                team2_score += 1
        
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
        print("NCAA Basketball Game Simulator v3".center(50))
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
        
        # Check for rivalry game
        if self.is_rivalry_game(team1, team2):
            print(f"\n⚡ {team1} vs {team2} is a RIVALRY GAME! Expect the unexpected! ⚡")
        
        print(f"\nSimulating {num_simulations} games between {team1} and {team2}...")
        print(f"Court: {'Neutral' if neutral_court else f'{team1} home'}")
        
        # Run simulations
        start_time = time.time()
        team1_wins = 0
        team2_wins = 0
        ties = 0
        team1_scores = []
        team2_scores = []
        team1_margins = []  # Track margins for confidence calculation
        
        for _ in range(num_simulations):
            score1, score2 = self.simulate_game(team1, team2, neutral_court)
            team1_scores.append(score1)
            team2_scores.append(score2)
            team1_margins.append(score1 - score2)
            
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
        margin_std = np.std(team1_margins)
        
        # Calculate confidence interval for the margin
        confidence_interval = stats.norm.interval(0.95, loc=margin, scale=margin_std/np.sqrt(num_simulations))
        
        # Find most common score
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
        
        # Win probability with confidence rating
        confidence_rating = min(5, max(1, int(abs(team1_wins - team2_wins) / (num_simulations * 0.1))))
        confidence_stars = "★" * confidence_rating + "☆" * (5 - confidence_rating)
        
        print("\nWin Probability:")
        print(f"{team1}: {team1_wins/num_simulations*100:.1f}%")
        print(f"{team2}: {team2_wins/num_simulations*100:.1f}%")
        print(f"Chance of Overtime: {ties/num_simulations*100:.1f}%")
        print(f"Prediction Confidence: {confidence_stars}")
        
        print("\nAverage Score:")
        print(f"{team1}: {team1_avg:.1f} ± {team1_std:.1f}")
        print(f"{team2}: {team2_avg:.1f} ± {team2_std:.1f}")
        print(f"Margin: {margin:.1f} points (95% confidence interval: {confidence_interval[0]:.1f} to {confidence_interval[1]:.1f})")
        
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
    simulator = NcaaGameSimulatorV3()
    simulator.run_simulation()

if __name__ == "__main__":
    main() 