"""
KenPom data manager for NCAA Basketball Game Simulator
Handles data retrieval and processing from KenPom.com
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from kenpompy.utils import login
import requests
from bs4 import BeautifulSoup
import time

class KenPomManager:
    def __init__(self, username=None, password=None):
        """
        Initialize KenPom data manager
        
        Args:
            username (str): KenPom.com username
            password (str): KenPom.com password
        """
        self.username = username
        self.password = password
        self.browser = None
        self.last_update = None
        self.team_stats = None
        
    def login(self):
        """
        Log in to KenPom.com
        
        Returns:
            bool: True if login successful, False otherwise
        """
        try:
            self.browser = login(self.username, self.password)
            return True
        except Exception as e:
            print(f"Login failed: {str(e)}")
            return False
            
    def load_team_stats(self, force_refresh=False):
        """
        Load team statistics from KenPom
        
        Args:
            force_refresh (bool): Whether to force refresh data from KenPom
            
        Returns:
            pd.DataFrame: Team statistics
        """
        # Check if we need to refresh the data
        if not force_refresh and self.team_stats is not None:
            # Check if data is less than 24 hours old
            if self.last_update and (datetime.now() - self.last_update).days < 1:
                return self.team_stats
        
        # Try to load from KenPom
        try:
            if not self.browser:
                if not self.login():
                    raise Exception("Failed to log in to KenPom")
            
            # Get team statistics page
            stats = self._scrape_team_stats()
            self.team_stats = stats
            self.last_update = datetime.now()
            
            # Save to cache
            self._save_to_cache()
            
            return stats
            
        except Exception as e:
            print(f"Error loading team stats: {str(e)}")
            # Try to load from cache
            return self._load_from_cache()
    
    def _scrape_team_stats(self):
        """
        Scrape team statistics from KenPom.com
        
        Returns:
            pd.DataFrame: Team statistics
        """
        # This is a placeholder for the actual scraping logic
        # In a real implementation, this would use the browser session
        # to navigate and parse the KenPom website
        
        # For now, return sample data
        return self._create_sample_data()
    
    def _create_sample_data(self):
        """
        Create sample team statistics for testing
        
        Returns:
            pd.DataFrame: Sample team statistics
        """
        teams = [
            "Gonzaga", "Baylor", "Michigan", "Illinois", "Iowa", 
            "Ohio State", "Alabama", "Houston", "Arkansas", "Purdue"
        ]
        
        data = {
            "Team": teams,
            "Conference": ["WCC", "Big 12", "Big Ten", "Big Ten", "Big Ten",
                         "Big Ten", "SEC", "AAC", "SEC", "Big Ten"],
            "AdjOE": np.random.uniform(110, 120, len(teams)),
            "AdjDE": np.random.uniform(90, 100, len(teams)),
            "Tempo-Adj": np.random.uniform(65, 75, len(teams)),
            "3P%": np.random.uniform(33, 40, len(teams)),
            "eFG%": np.random.uniform(50, 58, len(teams)),
            "TOV%": np.random.uniform(15, 20, len(teams)),
            "ORB%": np.random.uniform(25, 35, len(teams)),
            "FTR": np.random.uniform(30, 40, len(teams))
        }
        
        return pd.DataFrame(data)
    
    def _save_to_cache(self):
        """Save current team statistics to cache file"""
        if self.team_stats is not None:
            cache_dir = "cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_file = os.path.join(cache_dir, "team_stats_cache.csv")
            self.team_stats.to_csv(cache_file, index=False)
            
            # Save timestamp
            with open(os.path.join(cache_dir, "last_update.txt"), "w") as f:
                f.write(datetime.now().isoformat())
    
    def _load_from_cache(self):
        """
        Load team statistics from cache file
        
        Returns:
            pd.DataFrame: Cached team statistics
        """
        cache_file = os.path.join("cache", "team_stats_cache.csv")
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file)
        return None
    
    def get_team_stats(self, team_name):
        """
        Get statistics for a specific team
        
        Args:
            team_name (str): Name of the team
            
        Returns:
            pd.Series: Team statistics
        """
        if self.team_stats is None:
            self.load_team_stats()
            
        if self.team_stats is None:
            raise Exception("Failed to load team statistics")
            
        team_stats = self.team_stats[self.team_stats['Team'] == team_name]
        if len(team_stats) == 0:
            raise ValueError(f"Team '{team_name}' not found")
            
        return team_stats.iloc[0]
    
    def get_conference_stats(self, conference):
        """
        Get statistics for all teams in a conference
        
        Args:
            conference (str): Name of the conference
            
        Returns:
            pd.DataFrame: Conference statistics
        """
        if self.team_stats is None:
            self.load_team_stats()
            
        if self.team_stats is None:
            raise Exception("Failed to load team statistics")
            
        return self.team_stats[self.team_stats['Conference'] == conference]
    
    def get_ranking(self, team_name):
        """
        Get current KenPom ranking for a team
        
        Args:
            team_name (str): Name of the team
            
        Returns:
            int: Team's current ranking
        """
        if self.team_stats is None:
            self.load_team_stats()
            
        if self.team_stats is None:
            raise Exception("Failed to load team statistics")
            
        # Sort by adjusted efficiency margin (AdjOE - AdjDE)
        self.team_stats['AdjEM'] = self.team_stats['AdjOE'] - self.team_stats['AdjDE']
        rankings = self.team_stats.sort_values('AdjEM', ascending=False)
        
        team_rank = rankings.index[rankings['Team'] == team_name].tolist()
        if not team_rank:
            raise ValueError(f"Team '{team_name}' not found")
            
        return team_rank[0] + 1  # Convert 0-based index to 1-based ranking
    
    def get_matchup_history(self, team1, team2):
        """
        Get historical matchup data between two teams
        This is a placeholder that would normally fetch historical data
        
        Args:
            team1 (str): First team name
            team2 (str): Second team name
            
        Returns:
            dict: Historical matchup statistics
        """
        return {
            'games_played': 0,
            'team1_wins': 0,
            'team2_wins': 0,
            'avg_margin': 0,
            'last_meeting': None
        }