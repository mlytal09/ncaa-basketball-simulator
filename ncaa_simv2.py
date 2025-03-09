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
        
        # Load team statistics
        self.load_team_stats()
    
    def load_team_stats(self):
        """Load team statistics from CSV files"""
        try:
            # First try to load from the specified absolute path
            specific_path = r"C:\Users\mlyta\NCAA game simulator\stats\team_stats.csv"
            if os.path.exists(specific_path):
                self.team_stats = pd.read_csv(specific_path)
                print(f"Loading stats from {specific_path}")
            # Then try the relative paths as fallbacks
            elif os.path.exists("kenpom_stats.csv"):
                self.team_stats = pd.read_csv("kenpom_stats.csv")
                print("Loading stats from kenpom_stats.csv in main directory")
            elif os.path.exists(os.path.join(self.stats_dir, "kenpom_stats.csv")):
                self.team_stats = pd.read_csv(os.path.join(self.stats_dir, "kenpom_stats.csv"))
                print(f"Loading stats from {self.stats_dir}/kenpom_stats.csv")
            elif os.path.exists(os.path.join(self.stats_dir, "team_stats.csv")):
                self.team_stats = pd.read_csv(os.path.join(self.stats_dir, "team_stats.csv"))
                print(f"Loading stats from {self.stats_dir}/team_stats.csv")
            else:
                # If no file is found, use sample data instead of raising an error
                print("No stats file found, using sample data.")
                self._create_sample_data()
                return
            
            # Process the loaded data
            # If 'team_name' is in the DataFrame, set it as the index
            if 'team_name' in self.team_stats.columns:
                # Ensure team_name column contains strings
                self.team_stats['team_name'] = self.team_stats['team_name'].astype(str)
                # Convert team names to lowercase for case-insensitive matching
                self.team_stats['team_name_lower'] = self.team_stats['team_name'].str.lower()
                self.team_stats.set_index('team_name', inplace=True)
            else:
                # If there's no 'team_name' column, assume the first column is the team name
                first_col = self.team_stats.columns[0]
                self.team_stats.rename(columns={first_col: 'team_name'}, inplace=True)
                # Ensure team_name column contains strings
                self.team_stats['team_name'] = self.team_stats['team_name'].astype(str)
                self.team_stats['team_name_lower'] = self.team_stats['team_name'].str.lower()
                self.team_stats.set_index('team_name', inplace=True)
            
            print(f"Successfully loaded stats for {len(self.team_stats)} teams")
        except Exception as e:
            print(f"Error loading stats file: {e}")
            print("Using sample data for demonstration.")
            # Create sample data if file not found or there's an error
            self._create_sample_data()