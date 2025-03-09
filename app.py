import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ncaa_simv4 import NcaaGameSimulatorV3
import os

# Set page config
st.set_page_config(
    page_title="NCAA Basketball Game Simulator",
    page_icon="🏀",
    layout="wide"
)

# Title and description
st.title("NCAA Basketball Game Simulator 🏀")
st.markdown("""
Simulate NCAA basketball games using advanced analytics and team statistics.
This simulator accounts for team strength, home court advantage, rivalries, and more!
""")

# Initialize the simulator
@st.cache_resource
def load_simulator():
    simulator = NcaaGameSimulatorV3()
    # Ensure the stats directory exists
    os.makedirs("stats", exist_ok=True)
    return simulator

try:
    simulator = load_simulator()
except Exception as e:
    st.error(f"Error loading simulator: {str(e)}")
    st.error("Please ensure the team_stats.csv file is present in the stats directory.")
    st.stop()

# Create main columns
col1, col2 = st.columns(2)

with col1:
    # Game settings
    st.subheader("Game Settings")
    neutral_court = st.checkbox("Neutral Court Game")
    
    if not neutral_court:
        home_team = st.text_input("Home Team")
        away_team = st.text_input("Away Team")
    else:
        team1 = st.text_input("Team 1")
        team2 = st.text_input("Team 2")

    num_simulations = st.slider("Number of Simulations", 1000, 100000, 50000, 1000)

with col2:
    # Advanced settings
    st.subheader("Advanced Settings")
    show_histogram = st.checkbox("Show Score Distribution", value=True)
    show_details = st.checkbox("Show Detailed Statistics", value=True)

# Simulate button
if st.button("Run Simulation"):
    try:
        # Get team names based on court type
        if neutral_court:
            team1_name, team2_name = team1, team2
        else:
            team1_name, team2_name = home_team, away_team

        # Check if teams are entered
        if not team1_name or not team2_name:
            st.error("Please enter both team names.")
            st.stop()

        # Convert team names to lowercase for checking
        team1_lower = team1_name.lower()
        team2_lower = team2_name.lower()

        # Validate teams exist using team_stats index
        if team1_lower not in simulator.team_stats.index:
            similar_teams = simulator.find_similar_teams(team1_name)
            if similar_teams:
                st.warning(f"Team '{team1_name}' not found. Did you mean one of these? {', '.join(similar_teams)}")
            else:
                st.error(f"Team '{team1_name}' not found.")
            st.stop()

        if team2_lower not in simulator.team_stats.index:
            similar_teams = simulator.find_similar_teams(team2_name)
            if similar_teams:
                st.warning(f"Team '{team2_name}' not found. Did you mean one of these? {', '.join(similar_teams)}")
            else:
                st.error(f"Team '{team2_name}' not found.")
            st.stop()

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Initialize results storage
        team1_wins = 0
        team2_wins = 0
        overtime_games = 0
        team1_scores = []
        team2_scores = []

        # Run simulations
        for i in range(num_simulations):
            score1, score2, is_overtime = simulator.simulate_game(team1_name, team2_name, neutral_court)
            team1_scores.append(score1)
            team2_scores.append(score2)
            
            if score1 > score2:
                team1_wins += 1
            else:
                team2_wins += 1
                
            if is_overtime:
                overtime_games += 1

            # Update progress
            progress = (i + 1) / num_simulations
            progress_bar.progress(progress)
            status_text.text(f"Running simulation {i+1} of {num_simulations}")

        # Calculate statistics
        team1_win_pct = team1_wins / num_simulations * 100
        team2_win_pct = team2_wins / num_simulations * 100
        overtime_pct = overtime_games / num_simulations * 100

        team1_avg = np.mean(team1_scores)
        team2_avg = np.mean(team2_scores)
        team1_std = np.std(team1_scores)
        team2_std = np.std(team2_scores)

        # Clear progress bar and status
        progress_bar.empty()
        status_text.empty()

        # Display results
        st.subheader("Simulation Results")

        # Create three columns for results
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric("Win Probability", f"{max(team1_win_pct, team2_win_pct):.1f}%",
                     f"{team1_name if team1_win_pct > team2_win_pct else team2_name}")

        with res_col2:
            st.metric("Predicted Score", 
                     f"{team1_name}: {team1_avg:.1f}",
                     f"{team2_name}: {team2_avg:.1f}")

        with res_col3:
            st.metric("Overtime Chance", f"{overtime_pct:.1f}%")

        if show_details:
            st.subheader("Detailed Statistics")
            stats_col1, stats_col2 = st.columns(2)

            with stats_col1:
                st.write(f"**{team1_name} Statistics:**")
                st.write(f"Average Score: {team1_avg:.1f} ± {team1_std:.1f}")
                st.write(f"Win Percentage: {team1_win_pct:.1f}%")

            with stats_col2:
                st.write(f"**{team2_name} Statistics:**")
                st.write(f"Average Score: {team2_avg:.1f} ± {team2_std:.1f}")
                st.write(f"Win Percentage: {team2_win_pct:.1f}%")

        if show_histogram:
            st.subheader("Score Distribution")
            
            # Create histogram
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Team 1 histogram
            ax1.hist(team1_scores, bins=range(int(min(team1_scores))-5, int(max(team1_scores))+5),
                    color='blue', alpha=0.7)
            ax1.axvline(np.mean(team1_scores), color='red', linestyle='dashed', linewidth=2)
            ax1.set_xlabel('Points')
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'{team1_name} Score Distribution')

            # Team 2 histogram
            ax2.hist(team2_scores, bins=range(int(min(team2_scores))-5, int(max(team2_scores))+5),
                    color='green', alpha=0.7)
            ax2.axvline(np.mean(team2_scores), color='red', linestyle='dashed', linewidth=2)
            ax2.set_xlabel('Points')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'{team2_name} Score Distribution')

            plt.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Created with ❤️ by NCAA Basketball Game Simulator</p>
    <p>Data source: Team statistics from various sources</p>
</div>
""", unsafe_allow_html=True)