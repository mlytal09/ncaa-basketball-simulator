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
        
        # Load team stats - mandatory in V3
        self.load_team_stats()
        
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
            potential_team_columns = ['team', 'Team Name', 'TeamName', 'School']
            for col in potential_team_columns:
                if col in df.columns:
                    df = df.rename(columns={col: 'Team'})
                    print(f"Renamed column '{col}' to 'Team'")
                    break
            else:
                # If we can't find a team column, use the first column
                df = df.rename(columns={df.columns[0]: 'Team'})
                print(f"Using first column as 'Team'")
        
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
        Normalize team statistics for better comparison
        """
        # Define key statistics that should be normalized
        key_stats = [
            # Efficiency metrics
            "AdjO", "AdjD", "AdjEM", 
            # Four Factors
            "eFG%", "TOV%", "ORB%", "FTR",
            # Defensive stats
            "Def eFG%", "Def TOV%", "Def ORB%", "Def FTR",
            # SOS metrics
            "SOS AdjEM", "SOS OppO", "SOS OppD", "SOS AdjO", "SOS AdjD",
            # Offensive tendencies
            "3PA%", "A%", "2P%", "3P%", "FT%",
            # Defensive stats
            "Blk%", "Stl%", "Def 2P%", "Def 3P%", "Def FT%",
            # Height and tempo
            "Hgt", "Tempo"
        ]
        
        # Find numeric columns for normalization
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Remove columns we don't want to normalize
        exclude_cols = ['team_name_lower', 'Rk', 'W-L']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Combine key stats with numeric columns
        all_stats_to_normalize = list(set(numeric_cols + [stat for stat in key_stats if stat in df.columns]))
        
        # Track which key statistics are missing
        missing_key_stats = [stat for stat in key_stats if stat not in df.columns]
        if missing_key_stats:
            print(f"Warning: Missing key statistics: {', '.join(missing_key_stats)}")
        
        # Create a copy of the dataframe
        normalized_df = df.copy()
        
        # Normalize each column
        scaler = StandardScaler()
        if all_stats_to_normalize:
            # Fill NaN values with mean for normalization
            for col in all_stats_to_normalize:
                if col in normalized_df.columns:
                    normalized_df[col] = normalized_df[col].fillna(normalized_df[col].mean())
            
            # Perform normalization
            normalized_data = scaler.fit_transform(normalized_df[all_stats_to_normalize])
            
            # Add normalized columns back to dataframe
            for i, col in enumerate(all_stats_to_normalize):
                normalized_df[f"{col}_norm"] = normalized_data[:, i]
            
            print(f"Normalized {len(all_stats_to_normalize)} statistics: {', '.join(all_stats_to_normalize[:5])}...")
        
        return normalized_df

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
        """
        Check if a team exists in the dataset
        """
        if not team_name:
            return False
        
        team_lower = team_name.lower()
        
        # Direct match
        if team_lower in self.team_stats.index:
            return True
        
        # Partial match
        for idx in self.team_stats.index:
            if team_lower in idx:
                return True
        
        return False

    def find_similar_teams(self, team_name, threshold=0.6):
        """
        Find teams with similar names using fuzzy matching
        """
        if not team_name:
            return []
        
        team_lower = team_name.lower()
        similar_teams = []
        
        # Get all team names
        all_teams = [idx for idx in self.team_stats.index]
        
        # Check for partial matches first (more intuitive)
        for idx in all_teams:
            if team_lower in idx or idx in team_lower:
                # Get the actual team name with proper capitalization
                team_proper = self.team_stats.loc[idx]["Team"] if "Team" in self.team_stats.loc[idx] else idx
                similar_teams.append(team_proper)
        
        # If no partial matches, try fuzzy matching
        if not similar_teams:
            from difflib import SequenceMatcher
            
            for idx in all_teams:
                similarity = SequenceMatcher(None, team_lower, idx).ratio()
                if similarity >= threshold:
                    # Get the actual team name with proper capitalization
                    team_proper = self.team_stats.loc[idx]["Team"] if "Team" in self.team_stats.loc[idx] else idx
                    similar_teams.append(team_proper)
        
        # Return top 5 most similar teams
        return similar_teams[:5]
    
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
        Check if two teams are rivals
        """
        # Convert to lowercase for case-insensitive matching
        team1_lower = team1.lower()
        team2_lower = team2.lower()
        
        # Get team conferences if available
        team1_conf = None
        team2_conf = None
        
        try:
            if "Conference" in self.team_stats.loc[team1_lower]:
                team1_conf = self.team_stats.loc[team1_lower]["Conference"]
            if "Conference" in self.team_stats.loc[team2_lower]:
                team2_conf = self.team_stats.loc[team2_lower]["Conference"]
        except (KeyError, TypeError):
            # If we can't find the teams or conferences, just continue
            pass
        
        # Define known rivalries
        rivalries = [
            # ACC rivalries
            ("duke", "north carolina"),
            ("virginia", "virginia tech"),
            ("louisville", "kentucky"),
            # Big Ten rivalries
            ("michigan", "michigan state"),
            ("indiana", "purdue"),
            ("ohio state", "michigan"),
            # Big 12 rivalries
            ("kansas", "kansas state"),
            ("texas", "oklahoma"),
            ("iowa state", "iowa"),
            # SEC rivalries
            ("alabama", "auburn"),
            ("florida", "florida state"),
            ("kentucky", "tennessee"),
            # Pac-12 rivalries
            ("ucla", "usc"),
            ("arizona", "arizona state"),
            ("oregon", "oregon state"),
            # Big East rivalries
            ("villanova", "georgetown"),
            ("st. john's", "georgetown"),
            ("marquette", "wisconsin"),
            # Other major rivalries
            ("cincinnati", "xavier"),
            ("memphis", "tennessee"),
            ("byu", "utah"),
            ("gonzaga", "saint mary's"),
            ("creighton", "nebraska"),
            ("dayton", "xavier")
        ]
        
        # Check if teams are in the same conference (potential rivalry)
        same_conference = team1_conf and team2_conf and team1_conf == team2_conf
        
        # Check if teams are in a known rivalry
        for rival1, rival2 in rivalries:
            if (rival1 in team1_lower and rival2 in team2_lower) or (rival1 in team2_lower and rival2 in team1_lower):
                return True
        
        # Consider in-state matchups as potential rivalries
        # Extract state names from team names if possible
        states = [
            "alabama", "alaska", "arizona", "arkansas", "california", "colorado", 
            "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho", 
            "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", 
            "maine", "maryland", "massachusetts", "michigan", "minnesota", 
            "mississippi", "missouri", "montana", "nebraska", "nevada", 
            "new hampshire", "new jersey", "new mexico", "new york", "north carolina", 
            "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", 
            "rhode island", "south carolina", "south dakota", "tennessee", "texas", 
            "utah", "vermont", "virginia", "washington", "west virginia", 
            "wisconsin", "wyoming"
        ]
        
        # Check if both teams contain the same state name
        for state in states:
            if state in team1_lower and state in team2_lower:
                return True
        
        # Return True if teams are in the same conference (weaker signal)
        return same_conference
    
    def calculate_team_strength(self, team_stats):
        """
        Calculate team strength based on weighted statistics
        """
        strength = 0.0
        used_stats = []
        missing_stats = []
        
        # Apply weights to each statistic - prioritizing normalized versions if available
        for stat, weight in self.weights.items():
            # Try to use normalized version first
            norm_stat = f"{stat}_norm"
            if norm_stat in team_stats.index and not pd.isna(team_stats[norm_stat]):
                # For normalized stats, higher is always better
                norm_value = (team_stats[norm_stat] + 2) / 4  # Scale from roughly -2..2 to 0..1
                # Cap between 0 and 1
                norm_value = max(0, min(1, norm_value))
                strength += weight * norm_value
                used_stats.append(f"{stat} (normalized)")
            elif stat in team_stats.index and not pd.isna(team_stats[stat]):
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
                
                # Cap between 0 and 1
                norm_value = max(0, min(1, norm_value))
                strength += weight * norm_value
                used_stats.append(stat)
            else:
                missing_stats.append(stat)
        
        # Print debug info for the first calculation
        if not hasattr(self, '_printed_stats_debug'):
            print(f"Using {len(used_stats)} statistics for strength calculation")
            if missing_stats:
                print(f"Warning: Missing statistics: {', '.join(missing_stats)}")
            self._printed_stats_debug = True
        
        return strength
    
    def simulate_game(self, team1, team2, neutral_court=False):
        """
        Simulate a single game between two teams
        """
        # Get team stats
        try:
            team1_stats = self.team_stats.loc[team1.lower()]
            team2_stats = self.team_stats.loc[team2.lower()]
        except KeyError:
            # Try to find the team by searching for partial matches
            team1_found = False
            team2_found = False
            
            for team_name in self.team_stats.index:
                if team1.lower() in team_name:
                    team1 = team_name
                    team1_found = True
                if team2.lower() in team_name:
                    team2 = team_name
                    team2_found = True
            
            if not team1_found or not team2_found:
                raise ValueError(f"Could not find teams: {team1 if not team1_found else ''} {team2 if not team2_found else ''}")
            
            team1_stats = self.team_stats.loc[team1]
            team2_stats = self.team_stats.loc[team2]
        
        # Get team names with proper capitalization
        team1_name = team1_stats["Team"] if "Team" in team1_stats else team1
        team2_name = team2_stats["Team"] if "Team" in team2_stats else team2
        
        # Calculate team strengths
        team1_strength = self.calculate_team_strength(team1_stats)
        team2_strength = self.calculate_team_strength(team2_stats)
        
        # Apply home court advantage if not neutral
        if not neutral_court:
            if team1 == team2:
                raise ValueError("Home and away teams cannot be the same")
            
            # Apply home court advantage to team1 (home team)
            home_advantage = self.calculate_home_advantage(team1_stats)
            team1_strength *= (1 + home_advantage)
        
        # Determine if it's a rivalry game
        is_rivalry = self.is_rivalry_game(team1, team2)
        if is_rivalry:
            # Rivalry games are more unpredictable - reduce the strength gap
            avg_strength = (team1_strength + team2_strength) / 2
            team1_strength = team1_strength * 0.8 + avg_strength * 0.2
            team2_strength = team2_strength * 0.8 + avg_strength * 0.2
        
        # Calculate win probability
        total_strength = team1_strength + team2_strength
        team1_win_prob = team1_strength / total_strength
        team2_win_prob = team2_strength / total_strength
        
        # Safely get offensive and defensive stats with fallbacks
        def safe_get_stat(stats, stat_name, default_value):
            if stat_name in stats.index and not pd.isna(stats[stat_name]):
                return stats[stat_name]
            return default_value
        
        # Use adjusted defaults to create more modern, higher-scoring games
        # Modern NCAA average offensive efficiency is ~107-110
        team1_off = safe_get_stat(team1_stats, "AdjO", 107.0)
        team1_def = safe_get_stat(team1_stats, "AdjD", 97.0)
        team2_off = safe_get_stat(team2_stats, "AdjO", 107.0)
        team2_def = safe_get_stat(team2_stats, "AdjD", 97.0)
        
        # Get tempo with appropriate defaults (average NCAA tempo is ~68-70)
        team1_tempo = safe_get_stat(team1_stats, "Tempo", 68.5)
        team2_tempo = safe_get_stat(team2_stats, "Tempo", 68.5)
        
        # Small chance of an unusually paced game (fast or slow)
        pace_variation = random.random()
        if pace_variation > 0.9:  # 10% chance of unusual pace
            # Very fast or very slow game
            tempo_modifier = 1.15 if random.random() > 0.5 else 0.85
            team1_tempo *= tempo_modifier
            team2_tempo *= tempo_modifier
        
        # Calculate game tempo as weighted average of both teams
        # Teams with higher tempo exert more influence on game pace
        total_tempo = team1_tempo + team2_tempo
        team1_tempo_weight = team1_tempo / total_tempo
        team2_tempo_weight = team2_tempo / total_tempo
        
        # Calculate actual possessions with both teams' influence
        possessions = (team1_tempo * team1_tempo_weight) + (team2_tempo * team2_tempo_weight)
        
        # Random game-to-game variation in pace
        possessions *= (1 + np.random.normal(0, 0.06))  # 6% standard deviation
        
        # Adjust offensive and defensive ratings for matchup and game style
        # Offensive advantage over defense varies by matchup
        offense_advantage = 1.05  # Offense has slight advantage in modern college basketball
        
        # Calculate effective offensive and defensive ratings for the matchup
        team1_eff_off = team1_off * offense_advantage
        team2_eff_off = team2_off * offense_advantage
        
        # Calculate expected points per possession with proper weighting
        team1_ppp = (team1_eff_off / team2_def) * (105 / 100)
        team2_ppp = (team2_eff_off / team1_def) * (105 / 100)
        
        # Calculate raw scores based on possessions
        team1_raw_score = team1_ppp * possessions
        team2_raw_score = team2_ppp * possessions
        
        # Apply random variation - more in key games
        std_dev = 6.0  # Standard deviation (points)
        if is_rivalry:
            std_dev = 8.0  # More variation in rivalry games
        
        # Random team performance factors
        team1_score = np.random.normal(team1_raw_score, std_dev)
        team2_score = np.random.normal(team2_raw_score, std_dev)
        
        # Create realistic basketball scores
        team1_final = self.create_realistic_score(team1_score)
        team2_final = self.create_realistic_score(team2_score)
        
        # Check for overtime
        is_overtime = False
        if abs(team1_final - team2_final) <= 3:
            # Close game - chance of overtime (5%)
            if random.random() < 0.05:
                is_overtime = True
                # Add overtime points (typically 7-10 points per team in OT)
                ot_points1 = random.randint(5, 12)
                ot_points2 = random.randint(5, 12)
                
                # Slightly favor the team that was ahead
                if team1_final > team2_final:
                    ot_points1 += random.randint(0, 2)
                elif team2_final > team1_final:
                    ot_points2 += random.randint(0, 2)
                
                team1_final += ot_points1
                team2_final += ot_points2
        
        # Ensure no ties in final score
        if team1_final == team2_final:
            if random.random() < team1_win_prob:
                team1_final += 1
            else:
                team2_final += 1
        
        return team1_final, team2_final, is_overtime
    
    def create_realistic_score(self, raw_score):
        """
        Convert a raw score to a realistic basketball score with natural distribution
        """
        # Instead of a hard minimum, use a "soft minimum" approach
        # This gradually increases probability of higher scores when raw_score is low
        if raw_score < 65:
            # Apply a logarithmic transformation to low scores
            # This spreads out the scores below our target minimum instead of clustering them
            boost_factor = 1.0 - (raw_score / 65)  # 0 to 1 scale for how far below minimum
            boost_amount = boost_factor * 15  # Up to 15 points of boost
            
            # Add random boost based on how far below minimum (higher boost for lower scores)
            raw_score += boost_amount * random.random()
        
        # Apply normal distribution adjustment centered around realistic scores
        # This makes scores closer to realistic ranges more likely
        college_mean = 75  # Average NCAA score in recent seasons
        raw_score = raw_score * 0.85 + college_mean * 0.15  # 15% regression to mean
        
        # Round to integer, but keep decimal part for last-digit probability
        score_whole = int(raw_score)
        score_fraction = raw_score - score_whole
        
        # Basketball scores have specific patterns in their last digits
        # Adjust the last digit based on probability
        last_digit = score_whole % 10
        
        # Probability mapping for last digits in basketball scores
        digit_probs = {
            0: 0.15,  # Scores ending in 0 are common (free throws and even-numbered baskets)
            1: 0.08,
            2: 0.11,  # 2-point baskets
            3: 0.08,
            4: 0.09,
            5: 0.13,  # Scores ending in 5 are somewhat common (combo of FTs and 2pts)
            6: 0.09,
            7: 0.10,  # 3-point basket + 2-point scores
            8: 0.10,  # Multiple 2-point scores
            9: 0.07
        }
        
        # Use the fractional part to determine whether to adjust the last digit
        if random.random() < 0.7:  # 70% of scores will have this adjustment
            # Weight random choice based on digit probabilities
            new_last_digit = random.choices(
                list(digit_probs.keys()),
                weights=list(digit_probs.values())
            )[0]
            
            # Apply the new last digit
            score_whole = score_whole - last_digit + new_last_digit
        
        return score_whole
    
    def run_simulation(self, num_simulations=50000):
        """
        Run a full simulation of a game between two teams
        """
        # Get user input for teams
        neutral_court = input("Is this game on a neutral court? (yes/no): ").lower().startswith('y')
        
        if neutral_court:
            team1 = input("Enter the first team: ")
            team2 = input("Enter the second team: ")
        else:
            team1 = input("Enter the Home team: ")
            team2 = input("Enter the Away team: ")
        
        # Check if teams exist
        if not self.check_team_exists(team1):
            similar_teams = self.find_similar_teams(team1)
            if similar_teams:
                print(f"Team '{team1}' not found. Did you mean one of these?")
                for i, team in enumerate(similar_teams, 1):
                    print(f"{i}. {team}")
                choice = input("Enter the number of your choice (or press Enter to try again): ")
                if choice.isdigit() and 1 <= int(choice) <= len(similar_teams):
                    team1 = similar_teams[int(choice) - 1]
                else:
                    return self.run_simulation(num_simulations)
            else:
                print(f"Team '{team1}' not found and no similar teams found.")
                return self.run_simulation(num_simulations)
        
        if not self.check_team_exists(team2):
            similar_teams = self.find_similar_teams(team2)
            if similar_teams:
                print(f"Team '{team2}' not found. Did you mean one of these?")
                for i, team in enumerate(similar_teams, 1):
                    print(f"{i}. {team}")
                choice = input("Enter the number of your choice (or press Enter to try again): ")
                if choice.isdigit() and 1 <= int(choice) <= len(similar_teams):
                    team2 = similar_teams[int(choice) - 1]
                else:
                    return self.run_simulation(num_simulations)
            else:
                print(f"Team '{team2}' not found and no similar teams found.")
                return self.run_simulation(num_simulations)
        
        # Check if it's a rivalry game
        is_rivalry = self.is_rivalry_game(team1, team2)
        if is_rivalry:
            print(f"\n⚡ {team1} vs {team2} is a RIVALRY GAME! Expect the unexpected! ⚡\n")
        
        # Run the simulation
        print(f"Simulating {num_simulations} games between {team1} and {team2}...")
        if not neutral_court:
            print(f"Court: {team1} home")
        else:
            print("Court: Neutral")
        
        # Store results
        team1_wins = 0
        team2_wins = 0
        overtime_games = 0
        team1_scores = []
        team2_scores = []
        
        for _ in range(num_simulations):
            score1, score2, is_overtime = self.simulate_game(team1, team2, neutral_court)
            team1_scores.append(score1)
            team2_scores.append(score2)
            
            if score1 > score2:
                team1_wins += 1
            else:
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
        
        margin = team1_avg - team2_avg
        
        # Calculate 95% confidence interval for margin
        margin_std = np.std([s1 - s2 for s1, s2 in zip(team1_scores, team2_scores)])
        margin_error = 1.96 * margin_std / np.sqrt(num_simulations)
        margin_lower = margin - margin_error
        margin_upper = margin + margin_error
        
        # Find most common score
        score_counts = Counter(zip(team1_scores, team2_scores))
        most_common_score = score_counts.most_common(1)[0][0]
        
        # Calculate prediction confidence (1-5 stars)
        confidence_score = 0
        
        # Factor 1: Sample size
        if num_simulations >= 10000:
            confidence_score += 1
        
        # Factor 2: Win probability certainty
        win_certainty = abs(team1_win_pct - 50) / 50  # 0 to 1
        if win_certainty > 0.4:
            confidence_score += 1
        
        # Factor 3: Standard deviation of scores
        avg_std = (team1_std + team2_std) / 2
        if avg_std < 6:
            confidence_score += 1
        
        # Factor 4: Narrow confidence interval
        if abs(margin_upper - margin_lower) < 0.5:
            confidence_score += 1
        
        # Factor 5: Rivalry adjustment
        if not is_rivalry:
            confidence_score += 1
        
        # Convert to stars
        confidence_stars = "★" * confidence_score + "☆" * (5 - confidence_score)
        
        # Generate histogram
        histogram_path = self._generate_score_histogram(team1, team2, team1_scores, team2_scores)
        
        # Determine predicted winner and loser for display
        if team1_win_pct > team2_win_pct:
            predicted_winner = team1
            predicted_winner_score = round(team1_avg)
            predicted_loser = team2
            predicted_loser_score = round(team2_avg)
        else:
            predicted_winner = team2
            predicted_winner_score = round(team2_avg)
            predicted_loser = team1
            predicted_loser_score = round(team1_avg)
        
        # Ensure no tie in predicted score
        if predicted_winner_score == predicted_loser_score:
            predicted_winner_score += 1
        
        # Print results
        print("\n=== Simulation Results ===")
        print(f"Win Probability: {team1} {team1_win_pct:.1f}%, {team2} {team2_win_pct:.1f}%")
        print(f"Chance of Overtime: {overtime_pct:.1f}%")
        print(f"Prediction Confidence: {confidence_stars}")
        print(f"Average Score: {team1} {team1_avg:.1f} ± {team1_std:.1f}, {team2} {team2_avg:.1f} ± {team2_std:.1f}")
        print(f"Margin: {abs(margin):.1f} points (95% confidence interval: {margin_lower:.1f} to {margin_upper:.1f})")
        print(f"Most Common Score: {team1} {most_common_score[0]} - {team2} {most_common_score[1]}")
        
        # Predicted final score (rounded to integers)
        print(f"Predicted Final Score: {predicted_winner} {predicted_winner_score} - {predicted_loser} {predicted_loser_score}")
        
        if histogram_path:
            print(f"Score distribution histogram saved to {histogram_path}")
        
        return {
            "team1": team1,
            "team2": team2,
            "team1_win_pct": team1_win_pct,
            "team2_win_pct": team2_win_pct,
            "overtime_pct": overtime_pct,
            "team1_avg": team1_avg,
            "team2_avg": team2_avg,
            "team1_std": team1_std,
            "team2_std": team2_std,
            "margin": margin,
            "most_common_score": most_common_score,
            "histogram_path": histogram_path
        }
    
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