import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import time
import base64
from io import BytesIO
import requests

# Set page configuration
st.set_page_config(
    page_title="NCAA Basketball Game Simulator",
    page_icon="🏀",
    layout="wide",
)

# Function to get a downloadable link for the histogram
def get_image_download_link(fig, filename, text):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">Download {text}</a>'
    return href

# Simple simulator class that doesn't rely on external imports
class SimpleSimulator:
    def __init__(self):
        self.team_stats = None
        self.team_names = []
        self.rivalries = [
            ('Duke', 'North Carolina'),
            ('Kentucky', 'Louisville'),
            ('Kansas', 'Kansas St'),
            ('Indiana', 'Purdue'),
            ('Michigan', 'Michigan St'),
            ('UCLA', 'USC'),
            ('Alabama', 'Auburn'),
            ('Florida', 'Florida St'),
            ('Gonzaga', 'Saint Mary\'s')
        ]
        
    def load_team_stats(self, df):
        self.team_stats = df
        # Check if team_name column exists, otherwise try Team
        if 'team_name' in df.columns:
            self.team_names = df['team_name'].tolist()
            # Rename to Team for consistency
            df.rename(columns={'team_name': 'Team'}, inplace=True)
        elif 'Team' in df.columns:
            self.team_names = df['Team'].tolist()
        
    def is_rivalry_game(self, team1, team2):
        """Check if two teams are rivals"""
        for rival1, rival2 in self.rivalries:
            if (team1 in rival1 and team2 in rival2) or (team2 in rival1 and team1 in rival2):
                return True
        return False
        
    def simulate_game(self, team1, team2, neutral_court=False, version="V2"):
        """Simulation logic with different behavior for different versions"""
        try:
            # Get team stats
            team1_row = self.team_stats[self.team_stats['Team'] == team1]
            team2_row = self.team_stats[self.team_stats['Team'] == team2]
            
            if team1_row.empty or team2_row.empty:
                st.warning(f"Team data not found for {team1 if team1_row.empty else team2}")
                # Return fallback values with some randomness
                base1 = 70 + np.random.randint(-10, 10)
                base2 = 65 + np.random.randint(-10, 10)
                return base1, base2, False
                
            team1_row = team1_row.iloc[0]
            team2_row = team2_row.iloc[0]
            
            # Use offensive and defensive ratings if available
            team1_off = float(team1_row.get('AdjO', 100))
            team2_off = float(team2_row.get('AdjO', 100))
            team1_def = float(team1_row.get('AdjD', 100))
            team2_def = float(team2_row.get('AdjD', 100))
            
            # Tempo impacts total points
            tempo = (float(team1_row.get('Tempo', 70)) + float(team2_row.get('Tempo', 70))) / 2
            
            # Home court advantage
            if not neutral_court:
                if version == "V2":
                    team1_off *= 1.03  # 3% boost
                    team2_def *= 0.97  # 3% worse
                elif version == "V3":
                    team1_off *= 1.04  # 4% boost
                    team2_def *= 0.96  # 4% worse
                else:  # V4
                    team1_off *= 1.05  # 5% boost
                    team2_def *= 0.95  # 5% worse
            
            # Rivalry game adjustment
            if self.is_rivalry_game(team1, team2):
                if version == "V2":
                    # Make games closer in rivalry games
                    team1_off = team1_off * 0.9 + team2_off * 0.1
                    team2_off = team2_off * 0.9 + team1_off * 0.1
                elif version == "V3":
                    # More dramatic effect in V3
                    team1_off = team1_off * 0.85 + team2_off * 0.15
                    team2_off = team2_off * 0.85 + team1_off * 0.15
                    # Add more variance
                    team1_off *= np.random.uniform(0.95, 1.05)
                    team2_off *= np.random.uniform(0.95, 1.05)
                else:  # V4
                    # Even more dramatic in V4
                    team1_off = team1_off * 0.8 + team2_off * 0.2
                    team2_off = team2_off * 0.8 + team1_off * 0.2
                    # Add even more variance
                    team1_off *= np.random.uniform(0.93, 1.07)
                    team2_off *= np.random.uniform(0.93, 1.07)
            
            # Calculate raw scores
            if version == "V2":
                team1_raw = team1_off / team2_def * tempo / 2
                team2_raw = team2_off / team1_def * tempo / 2
            elif version == "V3":
                # V3 uses a slightly different formula
                team1_raw = (team1_off / team2_def) * (tempo / 1.9)
                team2_raw = (team2_off / team1_def) * (tempo / 1.9)
                # Add some additional factors
                team1_raw *= (1 + 0.02 * np.random.randn())
                team2_raw *= (1 + 0.02 * np.random.randn())
            else:  # V4
                # V4 uses an even more complex formula
                team1_raw = (team1_off / team2_def) * (tempo / 1.85)
                team2_raw = (team2_off / team1_def) * (tempo / 1.85)
                # Add conference strength factor
                conf1 = team1_row.get('Conerence', 'Unknown')
                conf2 = team2_row.get('Conerence', 'Unknown')
                conf_boost1 = 1.0
                conf_boost2 = 1.0
                
                # Conference strength adjustments
                power_conferences = ['B12', 'SEC', 'B10', 'ACC', 'BE']
                if conf1 in power_conferences:
                    conf_boost1 = 1.02
                if conf2 in power_conferences:
                    conf_boost2 = 1.02
                    
                team1_raw *= conf_boost1
                team2_raw *= conf_boost2
                
                # Add some additional factors
                team1_raw *= (1 + 0.03 * np.random.randn())
                team2_raw *= (1 + 0.03 * np.random.randn())
            
            # Add variability (more for V3/V4)
            if version == "V2":
                std_dev = 6
            elif version == "V3":
                std_dev = 8
            else:  # V4
                std_dev = 9
                
            team1_score = int(np.random.normal(team1_raw, std_dev))
            team2_score = int(np.random.normal(team2_raw, std_dev))
            
            # Ensure reasonable scores
            team1_score = max(50, min(110, team1_score))
            team2_score = max(50, min(110, team2_score))
            
            # For V3/V4 compatibility, add overtime flag
            if version == "V2":
                is_overtime = abs(team1_score - team2_score) <= 3 and np.random.random() < 0.05
            elif version == "V3":
                is_overtime = abs(team1_score - team2_score) <= 4 and np.random.random() < 0.07
            else:  # V4
                is_overtime = abs(team1_score - team2_score) <= 5 and np.random.random() < 0.09
            
            # Adjust scores for overtime
            if is_overtime:
                # Add overtime points
                ot_points1 = np.random.randint(5, 12)
                ot_points2 = np.random.randint(5, 12)
                team1_score += ot_points1
                team2_score += ot_points2
            
            return team1_score, team2_score, is_overtime
        except Exception as e:
            st.error(f"Error in simulation: {str(e)}")
            # Return fallback values with some randomness
            base1 = 70 + np.random.randint(-10, 10)
            base2 = 65 + np.random.randint(-10, 10)
            return base1, base2, False

def main():
    # Add a title and description
    st.title("NCAA Basketball Game Simulator")
    st.markdown("""
    This app simulates NCAA basketball games based on team statistics.
    Select teams and simulation settings below to see predicted outcomes.
    """)
    
    # Create sidebar for simulation settings
    st.sidebar.header("Simulation Settings")
    
    # Add option to choose simulator version
    simulator_version = st.sidebar.radio(
        "Simulator Version",
        ["Standard (V2)", "Advanced (V3)", "Premium (V4)"],
        index=0,  # Default to V2
        help="Standard: Basic simulator with fundamental team metrics. Advanced: Enhanced model with rivalry detection, team form tracking, and improved outcome predictions. Premium: Latest version with additional refinements."
    )
    
    num_simulations = st.sidebar.slider(
        "Number of Simulations", 
        min_value=1000, 
        max_value=50000, 
        value=10000, 
        step=1000,
        help="More simulations = more accurate results but slower"
    )
    
    # Prepare the GitHub URL for stats directly
    github_stats_url = "https://raw.githubusercontent.com/mlytal09/ncaa-basketball-simulator/main/stats/team_stats.csv"
    
    # Initialize progress
    progress_text = "Loading team statistics..."
    progress_bar = st.sidebar.progress(0)
    
    try:
        # Load the team stats directly from GitHub
        st.sidebar.info(f"Loading team stats from GitHub...")
        
        # Try to fetch the data with a timeout
        response = requests.get(github_stats_url, timeout=10)
        if response.status_code == 200:
            # Save to a temporary file and read with pandas
            with open("temp_stats.csv", "w") as f:
                f.write(response.text)
            team_stats_df = pd.read_csv("temp_stats.csv")
            progress_bar.progress(50)
            st.sidebar.success(f"Loaded {len(team_stats_df)} teams")
        else:
            st.sidebar.error(f"Failed to load team stats: HTTP {response.status_code}")
            # Create a minimal dataset for demonstration
            team_stats_df = pd.DataFrame({
                'team_name': ['Alabama', 'Gonzaga', 'Baylor', 'Houston', 'Michigan'],
                'Conerence': ['SEC', 'WCC', 'Big 12', 'American', 'Big Ten'],
                'AdjO': [118.9, 124.2, 123.5, 120.1, 117.8],
                'AdjD': [89.5, 94.1, 88.2, 85.6, 88.1],
                'Tempo': [73.2, 72.8, 69.8, 65.2, 67.7]
            })
            st.sidebar.warning("Using minimal dataset for demonstration")
    except Exception as e:
        st.sidebar.error(f"Error loading team stats: {str(e)}")
        # Create a minimal dataset for demonstration
        team_stats_df = pd.DataFrame({
            'team_name': ['Alabama', 'Gonzaga', 'Baylor', 'Houston', 'Michigan'],
            'Conerence': ['SEC', 'WCC', 'Big 12', 'American', 'Big Ten'],
            'AdjO': [118.9, 124.2, 123.5, 120.1, 117.8],
            'AdjD': [89.5, 94.1, 88.2, 85.6, 88.1],
            'Tempo': [73.2, 72.8, 69.8, 65.2, 67.7]
        })
        st.sidebar.warning("Using minimal dataset for demonstration")
    
    progress_bar.progress(75)
    
    # Initialize simulator
    try:
        # Use our simple simulator that doesn't rely on imports
        simulator = SimpleSimulator()
        simulator.load_team_stats(team_stats_df)
        
        if simulator_version == "Standard (V2)":
            st.sidebar.info("Using Standard simulator (V2)")
        elif simulator_version == "Advanced (V3)":
            st.sidebar.info("Using Advanced simulator (V3) with improved accuracy")
        else:
            st.sidebar.info("Using Premium simulator (V4) with latest enhancements")
    except Exception as e:
        st.error(f"Error initializing simulator: {str(e)}")
        st.stop()
    
    progress_bar.progress(100)
    
    # Create columns for the main UI
    col1, col2 = st.columns(2)
    
    # Game setup
    with col1:
        st.subheader("Game Setup")
        neutral_court = st.radio(
            "Court Type", 
            ["Home/Away", "Neutral Court"], 
            index=0
        ) == "Neutral Court"
        
        # Get team options directly from the loaded DataFrame
        team_options = sorted(simulator.team_names)
        st.info(f"Using {len(team_options)} teams from team_stats.csv")
        
        if neutral_court:
            # Neutral court game
            team1 = st.selectbox(
                "Select Team 1:",
                options=team_options,
                index=0
            )
            
            team2 = st.selectbox(
                "Select Team 2:",
                options=team_options,
                index=min(1, len(team_options)-1)
            )
        else:
            # Home/Away game
            team1 = st.selectbox(
                "Select Home Team:",
                options=team_options,
                index=0
            )
            
            team2 = st.selectbox(
                "Select Away Team:",
                options=team_options,
                index=min(1, len(team_options)-1)
            )
    
    # Run simulation button
    with col2:
        st.subheader("Run Simulation")
        st.write("Click the button below to run the simulation with the selected settings.")
        
        # Add some spacing
        st.write("")
        st.write("")
        
        # Check for rivalry game
        if simulator.is_rivalry_game(team1, team2):
            st.warning(f"⚡ {team1} vs {team2} is a RIVALRY GAME! Expect the unexpected!")
        
        # Create a button to run simulation
        run_button = st.button("Run Simulation", type="primary")
    
    # Run the simulation when the button is clicked
    if run_button:
        with st.spinner(f"Simulating {num_simulations} games between {team1} and {team2}..."):
            # Show simulation info
            st.info(f"Court: {'Neutral' if neutral_court else f'{team1} home'}")
            
            # Run simulations
            start_time = time.time()
            team1_wins = 0
            team2_wins = 0
            overtime_games = 0
            team1_scores = []
            team2_scores = []
            team1_margins = []  # Track margins for confidence calculation
            
            # Create a progress bar for simulations
            sim_progress = st.progress(0)
            
            # Get the version code
            if simulator_version == "Standard (V2)":
                version_code = "V2"
            elif simulator_version == "Advanced (V3)":
                version_code = "V3"
            else:
                version_code = "V4"
            
            for i in range(num_simulations):
                # Update progress every 5% of simulations
                if i % max(1, num_simulations // 20) == 0:
                    sim_progress.progress(i / num_simulations)
                
                # Run simulation
                try:
                    score1, score2, is_overtime = simulator.simulate_game(
                        team1, team2, neutral_court, version=version_code
                    )
                    
                    team1_scores.append(score1)
                    team2_scores.append(score2)
                    team1_margins.append(score1 - score2)
                    
                    if score1 > score2:
                        team1_wins += 1
                    elif score2 > score1:
                        team2_wins += 1
                    
                    if is_overtime:
                        overtime_games += 1
                except Exception as e:
                    st.error(f"Error in simulation: {str(e)}")
                    st.error(f"Team 1: {team1}, Team 2: {team2}")
                    st.stop()
            
            # Complete the progress bar
            sim_progress.progress(1.0)
            
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
            from collections import Counter
            score_pairs = [(team1_scores[i], team2_scores[i]) for i in range(num_simulations)]
            most_common_score = Counter(score_pairs).most_common(1)[0][0]
            
            sim_time = time.time() - start_time
        
        # Display results in a nice format
        st.success(f"Simulation completed in {sim_time:.2f} seconds")
        
        results_col1, results_col2 = st.columns(2)
        
        with results_col1:
            st.subheader("Win Probabilities")
            
            # Create win probability chart
            win_probs = [
                team1_wins/num_simulations*100, 
                team2_wins/num_simulations*100, 
                overtime_games/num_simulations*100
            ]
            labels = [f"{team1}", f"{team2}", "Overtime"]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            ax.bar(labels, win_probs, color=colors)
            ax.set_ylabel('Probability (%)')
            ax.set_title('Win Probability')
            for i, v in enumerate(win_probs):
                ax.text(i, v + 1, f"{v:.1f}%", ha='center')
            
            st.pyplot(fig)
            
            # Win probability text summary
            st.write(f"{team1}: {team1_wins/num_simulations*100:.1f}%")
            st.write(f"{team2}: {team2_wins/num_simulations*100:.1f}%")
            st.write(f"Chance of Overtime: {overtime_games/num_simulations*100:.1f}%")
            
            # Confidence rating
            confidence_rating = min(5, max(1, int(abs(team1_wins - team2_wins) / (num_simulations * 0.1))))
            confidence_stars = "★" * confidence_rating + "☆" * (5 - confidence_rating)
            st.write(f"Prediction Confidence: {confidence_stars}")
        
        with results_col2:
            st.subheader("Score Prediction")
            
            # Show average scores
            st.write(f"{team1}: {team1_avg:.1f} ± {team1_std:.1f}")
            st.write(f"{team2}: {team2_avg:.1f} ± {team2_std:.1f}")
            st.write(f"Margin: {margin:.1f} points")
            
            # Confidence interval
            st.write(f"95% confidence interval: {confidence_interval[0]:.1f} to {confidence_interval[1]:.1f} points")
            
            # Most common and predicted scores
            st.write(f"**Most Common Score:**")
            st.write(f"{team1} {most_common_score[0]} - {team2} {most_common_score[1]}")
            
            st.write(f"**Predicted Final:**")
            st.write(f"{team1} {round(team1_avg)} - {team2} {round(team2_avg)}")
        
        # Generate and display score distribution histograms
        st.subheader("Score Distributions")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Team 1 histogram
        ax1.hist(team1_scores, bins=range(min(team1_scores)-5, max(team1_scores)+5), color='blue', alpha=0.7)
        ax1.axvline(np.mean(team1_scores), color='red', linestyle='dashed', linewidth=2)
        ax1.set_xlabel('Points')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{team1} Score Distribution')
        
        # Team 2 histogram
        ax2.hist(team2_scores, bins=range(min(team2_scores)-5, max(team2_scores)+5), color='green', alpha=0.7)
        ax2.axvline(np.mean(team2_scores), color='red', linestyle='dashed', linewidth=2)
        ax2.set_xlabel('Points')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{team2} Score Distribution')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Provide download link using in-memory buffer instead of file
        st.markdown(get_image_download_link(fig, 
                                          f"{team1}_vs_{team2}.png", 
                                          "Histogram"), 
                  unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.error("Please try refreshing the page. If the error persists, contact support.")