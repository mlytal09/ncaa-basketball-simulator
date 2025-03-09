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
        
        Args:
            stats_dir (str): Directory containing team statistics CSV files
        """
        self.stats_dir = stats_dir
        self.team_stats = None
        self._printed_stats_debug = False
        
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
        
        # Define conference tiers for strength adjustment
        self.conference_tiers = {
            'Big 12': 1,
            'Big East': 1,
            'Big Ten': 1,
            'SEC': 1,
            'ACC': 1,
            'Pac-12': 2,
            'American': 2,
            'Mountain West': 2,
            'Atlantic 10': 2,
            'WCC': 2,
            'MVC': 3,
            'C-USA': 3,
            'MAC': 3,
            'Sun Belt': 3,
            'Horizon': 3,
            'CAA': 3,
            'WAC': 3,
            'SoCon': 3,
            'Ivy': 3,
            'Big West': 4,
            'Big Sky': 4,
            'America East': 4,
            'ASUN': 4,
            'Big South': 4,
            'MAAC': 4,
            'NEC': 4,
            'OVC': 4,
            'Patriot': 4,
            'Southland': 4,
            'Summit': 4,
            'SWAC': 5,
            'MEAC': 5
        }
        
        # Define rivalry pairs
        self.rivalries = [
            ('Duke', 'North Carolina'),
            ('Kentucky', 'Louisville'),
            ('Kansas', 'Kansas St'),
            ('Indiana', 'Purdue'),
            ('Michigan', 'Michigan St'),
            ('UCLA', 'USC'),
            ('Xavier', 'Cincinnati'),
            ('Villanova', 'Georgetown'),
            ('Syracuse', 'Georgetown'),
            ('Oklahoma', 'Oklahoma St'),
            ('Alabama', 'Auburn'),
            ('Florida', 'Florida St'),
            ('Illinois', 'Missouri'),
            ('Iowa', 'Iowa St'),
            ('BYU', 'Utah'),
            ('Gonzaga', 'Saint Mary\'s'),
            ('Arizona', 'Arizona St'),
            ('Texas', 'Texas A&M'),
            ('Ohio St', 'Michigan'),
            ('Marquette', 'Wisconsin'),
            ('Memphis', 'Tennessee'),
            ('Creighton', 'Nebraska'),
            ('Baylor', 'Texas'),
            ('UConn', 'Syracuse'),
            ('Pitt', 'West Virginia'),
            ('Temple', 'Penn'),
            ('VCU', 'Richmond'),
            ('Utah', 'Utah St'),
            ('Washington', 'Washington St'),
            ('Oregon', 'Oregon St'),
            ('New Mexico', 'New Mexico St'),
            ('Nevada', 'UNLV'),
            ('SMU', 'TCU'),
            ('Dayton', 'Xavier'),
            ('Butler', 'Xavier'),
            ('Saint Louis', 'SIU Edwardsville'),
            ('Loyola Chicago', 'DePaul'),
            ('Belmont', 'Lipscomb'),
            ('Murray St', 'Western Kentucky'),
            ('Wichita St', 'Kansas'),
            ('Valparaiso', 'Butler'),
            ('Drake', 'Northern Iowa'),
            ('Evansville', 'Indiana St'),
            ('Akron', 'Kent St'),
            ('Buffalo', 'Canisius'),
            ('Toledo', 'Bowling Green'),
            ('Ohio', 'Miami OH'),
            ('Davidson', 'Furman'),
            ('Wofford', 'Furman'),
            ('ETSU', 'Chattanooga'),
            ('Mercer', 'Samford'),
            ('Iona', 'Manhattan'),
            ('Monmouth', 'Saint Peter\'s'),
            ('Fairfield', 'Quinnipiac'),
            ('Rider', 'Siena'),
            ('Vermont', 'Albany'),
            ('UMBC', 'Stony Brook'),
            ('Binghamton', 'Albany'),
            ('Hartford', 'Maine'),
            ('Northeastern', 'Hofstra'),
            ('Charleston', 'UNCW'),
            ('Towson', 'Delaware'),
            ('Drexel', 'Delaware'),
            ('James Madison', 'William & Mary'),
            ('Elon', 'UNCW'),
            ('Weber St', 'Montana'),
            ('Eastern Washington', 'Idaho'),
            ('Northern Colorado', 'Montana St'),
            ('Portland St', 'Idaho St'),
            ('Sacramento St', 'UC Davis'),
            ('Cal Poly', 'UC Santa Barbara'),
            ('UC Irvine', 'Long Beach St'),
            ('Hawaii', 'Cal St Fullerton'),
            ('UC Riverside', 'UC San Diego'),
            ('North Dakota St', 'South Dakota St'),
            ('Oral Roberts', 'South Dakota'),
            ('UMKC', 'Western Illinois'),
            ('Denver', 'Omaha'),
            ('Purdue Fort Wayne', 'Cleveland St'),
            ('Northern Kentucky', 'Wright St'),
            ('Milwaukee', 'Green Bay'),
            ('Detroit Mercy', 'Oakland'),
            ('Youngstown St', 'Robert Morris'),
            ('IUPUI', 'UIC'),
            ('Stephen F. Austin', 'Sam Houston St'),
            ('Abilene Christian', 'Incarnate Word'),
            ('Nicholls St', 'SE Louisiana'),
            ('New Orleans', 'McNeese St'),
            ('Texas A&M-CC', 'Lamar'),
            ('Central Arkansas', 'Northwestern St'),
            ('Grand Canyon', 'New Mexico St'),
            ('Seattle', 'Utah Valley'),
            ('Cal Baptist', 'Dixie St'),
            ('UT Rio Grande Valley', 'Tarleton St'),
            ('Chicago St', 'UTRGV'),
            ('Southern Utah', 'Northern Arizona'),
            ('Kennesaw St', 'North Florida'),
            ('Liberty', 'Jacksonville'),
            ('FGCU', 'Stetson'),
            ('Lipscomb', 'North Alabama'),
            ('Bellarmine', 'Eastern Kentucky'),
            ('Central Arkansas', 'Jacksonville St'),
            ('Winthrop', 'High Point'),
            ('Radford', 'UNC Asheville'),
            ('Gardner-Webb', 'Campbell'),
            ('Presbyterian', 'Charleston Southern'),
            ('USC Upstate', 'Longwood'),
            ('Hampton', 'Norfolk St'),
            ('Morgan St', 'Coppin St'),
            ('NC Central', 'NC A&T'),
            ('Howard', 'Delaware St'),
            ('South Carolina St', 'Bethune-Cookman'),
            ('Florida A&M', 'Maryland Eastern Shore'),
            ('Alcorn St', 'Jackson St'),
            ('Southern', 'Grambling'),
            ('Prairie View', 'Texas Southern'),
            ('Alabama St', 'Alabama A&M'),
            ('Mississippi Valley St', 'Arkansas-Pine Bluff'),
            ('Lehigh', 'Lafayette'),
            ('Bucknell', 'Holy Cross'),
            ('Colgate', 'Army'),
            ('Navy', 'American'),
            ('Boston U', 'Loyola MD'),
            ('Fairleigh Dickinson', 'Wagner'),
            ('Bryant', 'Mount St. Mary\'s'),
            ('St. Francis PA', 'Robert Morris'),
            ('St. Francis NY', 'LIU'),
            ('Sacred Heart', 'Central Connecticut'),
            ('Merrimack', 'Bryant'),
            ('Belmont', 'Tennessee St'),
            ('Austin Peay', 'Murray St'),
            ('Eastern Illinois', 'SIU Edwardsville'),
            ('Eastern Kentucky', 'Morehead St'),
            ('Tennessee Tech', 'Jacksonville St'),
            ('UT Martin', 'Southeast Missouri St'),
            ('Army', 'Navy'),
            ('Air Force', 'Army'),
            ('Navy', 'Air Force')
        ]
        
        # Load team statistics
        self.load_team_stats()
    
    def load_team_stats(self):
        """
        Load team statistics from CSV file
        """
        # Try to load from the specified absolute path first
        specific_path = r"C:\Users\mlyta\NCAA game simulator\stats\team_stats.csv"
        if os.path.exists(specific_path):
            stats_file = specific_path
        else:
            # Fall back to relative paths
            stats_file = os.path.join(self.stats_dir, "team_stats.csv")
            if not os.path.exists(stats_file):
                stats_file = os.path.join(self.stats_dir, "kenpom_stats.csv")
                if not os.path.exists(stats_file):
                    raise FileNotFoundError(f"Could not find team stats file in {self.stats_dir} or at {specific_path}")
        
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
        
        # Save original team names before setting index
        self.team_names = self.team_stats['Team'].tolist()
        
        # Set the team name as index for faster lookups, but keep Team column accessible
        self.team_stats_indexed = self.team_stats.set_index('team_name_lower')
        
        # Create a lookup dict for team name to team name lower
        self.team_name_to_lower = {
            team: team.lower() for team in self.team_names
        }
        
        print(f"Successfully loaded stats for {len(self.team_stats)} teams")
        return self.team_stats
    
    def normalize_team_stats(self, df):
        """
        Normalize team statistics to ensure consistent data format
        
        Args:
            df (DataFrame): The loaded team statistics DataFrame
            
        Returns:
            DataFrame: Normalized team statistics
        """
        # Create a copy to avoid modifying the original
        df_norm = df.copy()
        
        # Make sure team names are preserved
        if 'Team' in df_norm.columns:
            team_col = 'Team'
        elif 'team_name' in df_norm.columns:
            team_col = 'team_name'
        else:
            # If no team column, use the first column
            team_col = df_norm.columns[0]
            df_norm.rename(columns={team_col: 'Team'}, inplace=True)
            team_col = 'Team'
        
        # Remove any potential duplicate teams
        df_norm.drop_duplicates(subset=[team_col], keep='first', inplace=True)
        
        # Check for essential columns and set defaults if missing
        essential_stats = [
            'AdjO', 'AdjD', 'Tempo', 'eFG%', 'TOV%', 'ORB%', 'FTR',
            'Blk%', 'Stl%', '3PA%', 'A%'
        ]
        
        for stat in essential_stats:
            if stat not in df_norm.columns:
                # Use reasonable defaults for missing columns
                if stat == 'AdjO' or stat == 'AdjD':
                    df_norm[stat] = 100.0  # Average efficiency
                elif stat == 'Tempo':
                    df_norm[stat] = 68.0   # Average tempo
                elif stat in ['eFG%', 'TOV%', 'ORB%', 'FTR', '3PA%']:
                    df_norm[stat] = 0.5    # Mid-range value
                elif stat in ['Blk%', 'Stl%']:
                    df_norm[stat] = 0.1    # Typical values
                elif stat == 'A%':
                    df_norm[stat] = 0.5    # Mid-range value
                print(f"Created default column for missing stat: {stat}")
        
        # Ensure team_name_lower column exists
        if 'team_name_lower' not in df_norm.columns:
            df_norm['team_name_lower'] = df_norm[team_col].str.lower()
        
        # Convert percentage strings to floats if needed
        for col in df_norm.columns:
            if col != team_col and col != 'team_name_lower' and col != 'Conference':
                # Try to convert to numeric, errors='coerce' will convert failures to NaN
                df_norm[col] = pd.to_numeric(df_norm[col], errors='coerce')
                
                # If the column has '%' in its name but values are not between 0-1,
                # divide by 100 to normalize
                if '%' in col and df_norm[col].max() > 1:
                    df_norm[col] = df_norm[col] / 100.0
        
        # Replace NaN values with reasonable defaults
        for col in df_norm.columns:
            if col not in [team_col, 'team_name_lower', 'Conference']:
                if df_norm[col].isna().any():
                    # Choose sensible defaults based on the statistic
                    if col in ['AdjO', 'AdjD']:
                        df_norm[col].fillna(100.0, inplace=True)
                    elif '%' in col:
                        df_norm[col].fillna(0.5, inplace=True)
                    else:
                        # For other metrics, use the column mean
                        col_mean = df_norm[col].mean()
                        if pd.isna(col_mean):  # If mean is also NaN, use a reasonable default
                            col_mean = 0.5
                        df_norm[col].fillna(col_mean, inplace=True)
        
        return df_norm
    
    def get_conference_tier(self, conference):
        """
        Get the tier ranking for a conference
        
        Args:
            conference (str): Conference name
            
        Returns:
            int: Tier ranking (1-5, with 1 being highest)
        """
        if conference in self.conference_tiers:
            return self.conference_tiers[conference]
        else:
            # Default to mid-tier if conference not found
            return 3

    def calculate_home_advantage(self, home_team_stats):
        """
        Calculate home court advantage factor
        
        Args:
            home_team_stats (Series): Statistics for the home team
            
        Returns:
            float: Home court advantage factor (0.0-0.1)
        """
        # Base home court advantage (3-4%)
        advantage = 0.035
        
        # Adjust based on conference if available
        if 'Conference' in home_team_stats:
            conference = home_team_stats['Conference']
            conference_tier = self.get_conference_tier(conference)
            
            # Higher tier conferences tend to have stronger home court advantage
            tier_adjustment = (6 - conference_tier) * 0.005  # 0.005 to 0.025
            advantage += tier_adjustment
        
        # Add random variation (±1%)
        advantage += np.random.normal(0, 0.01)
        
        # Ensure advantage is within reasonable bounds
        advantage = max(0.02, min(0.08, advantage))
        
        return advantage
    
    def check_team_exists(self, team_name):
        """
        Check if a team exists in the loaded statistics
        
        Args:
            team_name (str): Name of the team to check
            
        Returns:
            bool: True if team exists, False otherwise
        """
        if self.team_stats is None:
            self.load_team_stats()
        
        # Check in team_names list (original case)
        if team_name in self.team_names:
            return True
        
        # Try lowercase match in indexed dataframe
        if team_name.lower() in self.team_stats_indexed.index:
            return True
        
        # Try partial match
        for team in self.team_names:
            if team_name.lower() in team.lower():
                return True
        
        return False
    
    def find_similar_teams(self, team_name, threshold=0.6):
        """
        Find teams with similar names
        
        Args:
            team_name (str): Name to search for
            threshold (float): Similarity threshold (0-1)
            
        Returns:
            list: List of similar team names
        """
        if self.team_stats is None:
            self.load_team_stats()
        
        team_name = team_name.lower()
        similar_teams = []
        
        # First try direct substring match
        for team in self.team_names:
            if team_name in team.lower() or team.lower() in team_name:
                similar_teams.append(team)
        
        # If no direct matches, try fuzzy matching
        if not similar_teams:
            for team in self.team_names:
                similarity = difflib.SequenceMatcher(None, team_name, team.lower()).ratio()
                if similarity >= threshold:
                    similar_teams.append(team)
        
        # Sort by similarity (most similar first)
        similar_teams.sort(key=lambda x: difflib.SequenceMatcher(None, team_name, x.lower()).ratio(), reverse=True)
        
        return similar_teams
    
    def get_team_form(self, team_name):
        """
        Get a team's current form (momentum)
        
        Args:
            team_name (str): Name of the team
            
        Returns:
            float: Form factor (-1.0 to 1.0, with 1.0 being excellent form)
        """
        # In a real implementation, this would use recent game results
        # For now, generate a random form value with slight bias toward average
        form = np.random.normal(0, 0.5)
        
        # Clamp to reasonable range
        form = max(-1.0, min(1.0, form))
        
        return form
    
    def is_rivalry_game(self, team1, team2):
        """
        Check if two teams are rivals
        
        Args:
            team1 (str): First team name
            team2 (str): Second team name
            
        Returns:
            bool: True if teams are rivals, False otherwise
        """
        # Normalize team names
        team1 = team1.strip()
        team2 = team2.strip()
        
        # Check if teams are in the rivalry list (in either order)
        if (team1, team2) in self.rivalries or (team2, team1) in self.rivalries:
            return True
        
        # Check for partial matches
        for rivalry in self.rivalries:
            # Check if both teams partially match a rivalry pair
            t1_match = (team1.lower() in rivalry[0].lower() or rivalry[0].lower() in team1.lower())
            t2_match = (team2.lower() in rivalry[1].lower() or rivalry[1].lower() in team2.lower())
            
            if (t1_match and t2_match):
                return True
            
            # Check the reverse order
            t1_match = (team1.lower() in rivalry[1].lower() or rivalry[1].lower() in team1.lower())
            t2_match = (team2.lower() in rivalry[0].lower() or rivalry[0].lower() in team2.lower())
            
            if (t1_match and t2_match):
                return True
        
        # Also check conference rivals for teams in the same conference
        try:
            # Get team stats by converting to lowercase first
            team1_lower = team1.lower()
            team2_lower = team2.lower()
            
            if team1_lower in self.team_stats_indexed.index and team2_lower in self.team_stats_indexed.index:
                team1_stats = self.team_stats_indexed.loc[team1_lower]
                team2_stats = self.team_stats_indexed.loc[team2_lower]
                
                if 'Conference' in team1_stats and 'Conference' in team2_stats:
                    if team1_stats['Conference'] == team2_stats['Conference']:
                        # Same conference - calculate chance of being rivals
                        conference_tier = self.get_conference_tier(team1_stats['Conference'])
                        # Lower tier conferences have stronger internal rivalries
                        rivalry_chance = 0.2 - (conference_tier * 0.03)
                        
                        # Random chance of being conference rivals
                        if random.random() < rivalry_chance:
                            return True
        except:
            # If there's an error accessing team stats, just continue
            pass
        
        return False
    
    def calculate_team_strength(self, team_stats):
        """
        Calculate overall team strength based on weighted statistics
        
        Args:
            team_stats (Series): Team statistics
            
        Returns:
            float: Overall team strength value
        """
        strength = 0.0
        valid_stats = 0
        missing_stats = []
        
        for stat, weight in self.weights.items():
            if stat in team_stats and not pd.isna(team_stats[stat]):
                # For AdjO and AdjD, use directly (they're already on a common scale)
                if stat in ['AdjO', 'AdjD']:
                    value = team_stats[stat]
                elif stat == 'AdjD':
                    # For defensive rating, lower is better, so invert the contribution
                    value = 200 - team_stats[stat]
                # For percentage stats, multiply by 100 to get on similar scale
                elif '%' in stat:
                    value = team_stats[stat] * 100
                else:
                    value = team_stats[stat]
                
                # Add weighted contribution to strength
                strength += value * weight
                valid_stats += 1
            else:
                missing_stats.append(stat)
        
        # Adjust if not all stats were available
        if valid_stats < len(self.weights):
            strength = strength * (len(self.weights) / valid_stats)
        
        # Only print missing stats debug info once
        if missing_stats and not hasattr(self, '_printed_stats_debug'):
            if len(missing_stats) > 5:
                print(f"Warning: Missing many statistics: {len(missing_stats)} stats")
            else:
                print(f"Warning: Missing statistics: {', '.join(missing_stats)}")
            self._printed_stats_debug = True
        
        return strength
    
    def simulate_game(self, team1, team2, neutral_court=False):
        """
        Simulate a single game between two teams
        """
        # Get team stats
        try:
            # Try to find the team by exact match first
            team1_lower = team1.lower()
            team2_lower = team2.lower()
            
            if team1_lower in self.team_stats_indexed.index:
                team1_stats = self.team_stats_indexed.loc[team1_lower]
            else:
                # Try to find by finding a similar team name
                similar_teams = self.find_similar_teams(team1)
                if similar_teams:
                    team1 = similar_teams[0]
                    team1_lower = team1.lower()
                    team1_stats = self.team_stats_indexed.loc[team1_lower]
                else:
                    raise ValueError(f"Team '{team1}' not found in stats database")
                
            if team2_lower in self.team_stats_indexed.index:
                team2_stats = self.team_stats_indexed.loc[team2_lower]
            else:
                # Try to find by finding a similar team name
                similar_teams = self.find_similar_teams(team2)
                if similar_teams:
                    team2 = similar_teams[0]
                    team2_lower = team2.lower()
                    team2_stats = self.team_stats_indexed.loc[team2_lower]
                else:
                    raise ValueError(f"Team '{team2}' not found in stats database")
        except Exception as e:
            # Try to find the team by searching for partial matches
            print(f"Error finding teams: {e}")
            team1_found = False
            team2_found = False
            
            for team_name in self.team_names:
                if team1.lower() in team_name.lower():
                    team1 = team_name
                    team1_found = True
                    break
            
            for team_name in self.team_names:
                if team2.lower() in team_name.lower():
                    team2 = team_name
                    team2_found = True
                    break
            
            if not team1_found:
                raise ValueError(f"Team '{team1}' not found in stats database")
            
            if not team2_found:
                raise ValueError(f"Team '{team2}' not found in stats database")
            
            # Try again with the found team names
            team1_lower = team1.lower()
            team2_lower = team2.lower()
            team1_stats = self.team_stats_indexed.loc[team1_lower]
            team2_stats = self.team_stats_indexed.loc[team2_lower]
        
        # Calculate team strengths
        team1_strength = self.calculate_team_strength(team1_stats)
        team2_strength = self.calculate_team_strength(team2_stats)
        
        # Apply home court advantage if not neutral court
        if not neutral_court:
            home_advantage = self.calculate_home_advantage(team1_stats)
            team1_strength *= (1 + home_advantage)
        
        # Check if this is a rivalry game
        if self.is_rivalry_game(team1, team2):
            # Rivalry games are more unpredictable - reduce the gap between teams
            # This gives the underdog a better chance
            avg_strength = (team1_strength + team2_strength) / 2
            team1_strength = team1_strength * 0.7 + avg_strength * 0.3
            team2_strength = team2_strength * 0.7 + avg_strength * 0.3
        
        # Get team form (recent performance)
        team1_form = self.get_team_form(team1)
        team2_form = self.get_team_form(team2)
        
        # Apply form adjustment (up to 5% boost/penalty)
        team1_strength *= (1 + team1_form * 0.05)
        team2_strength *= (1 + team2_form * 0.05)
        
        # Calculate win probability based on team strengths
        # Using a sigmoid function to convert strength difference to probability
        strength_diff = team1_strength - team2_strength
        team1_win_prob = 1 / (1 + np.exp(-strength_diff * 0.1))
        
        # Determine tempo (possessions per game)
        # Use the average of both teams' tempos, with a slight lean toward the home team
        if not neutral_court:
            tempo_weight = 0.55  # Home team has slightly more control over tempo
        else:
            tempo_weight = 0.5  # Equal influence on neutral court
        
        # Safe get for tempo values
        def safe_get_stat(stats, stat_name, default_value):
            if stat_name in stats and not pd.isna(stats[stat_name]):
                return stats[stat_name]
            return default_value
        
        team1_tempo = safe_get_stat(team1_stats, 'Tempo', 68.0)
        team2_tempo = safe_get_stat(team2_stats, 'Tempo', 68.0)
        
        # Calculate game tempo with random variation
        base_tempo = team1_tempo * tempo_weight + team2_tempo * (1 - tempo_weight)
        game_tempo = np.random.normal(base_tempo, 2.0)  # Add some random variation
        
        # Ensure tempo is within reasonable bounds
        game_tempo = max(60, min(80, game_tempo))
        
        # Calculate offensive and defensive ratings
        team1_off = safe_get_stat(team1_stats, 'AdjO', 100.0)
        team1_def = safe_get_stat(team1_stats, 'AdjD', 100.0)
        team2_off = safe_get_stat(team2_stats, 'AdjO', 100.0)
        team2_def = safe_get_stat(team2_stats, 'AdjD', 100.0)
        
        # Calculate expected points per possession
        team1_ppp = team1_off / 100.0
        team2_ppp = team2_off / 100.0
        
        # Adjust for opponent defense
        team1_ppp *= (100.0 / team2_def)
        team2_ppp *= (100.0 / team1_def)
        
        # Calculate raw scores based on tempo and efficiency
        team1_raw_score = team1_ppp * game_tempo
        team2_raw_score = team2_ppp * game_tempo
        
        # Add random variation based on win probability
        # More variation for closer matchups
        evenness_factor = 4.0 * team1_win_prob * (1 - team1_win_prob)  # Peaks at 0.5 probability
        std_dev = 6.0 + evenness_factor * 4.0  # Between 6-10 points of standard deviation
        
        # Random team performance factors with scaled variance
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
        if raw_score < 60:
            # Apply a logarithmic transformation to low scores
            # This spreads out the scores below our target minimum instead of clustering them
            boost_factor = 1.0 - (raw_score / 60)  # 0 to 1 scale for how far below minimum
            boost_amount = boost_factor * 12  # Up to 12 points of boost (reduced from 15)
            
            # Add random boost based on how far below minimum (higher boost for lower scores)
            raw_score += boost_amount * random.random()
        
        # Apply normal distribution adjustment centered around realistic scores
        # NCAA average score is around 71-73 in recent seasons
        college_mean = 72  # More moderate average (reduced from 75)
        raw_score = raw_score * 0.8 + college_mean * 0.2  # 20% regression to mean (increased from 15%)
        
        # Round to integer, but keep decimal part for last-digit probability
        score_whole = int(raw_score)
        score_fraction = raw_score - score_whole
        
        # Use decimal part to potentially round up (creates more natural distribution of final digits)
        if random.random() < score_fraction:
            score_whole += 1
        
        return score_whole