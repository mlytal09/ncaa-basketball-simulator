import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib.util
import sys
from pathlib import Path

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

# Function to load a module from file path
def load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Function to initialize the simulator based on version
@st.cache_resource
def load_simulator(version):
    # Map version names to file paths and class names
    version_map = {
        "Version 2": {
            "file": "ncaa_simv2.py",
            "class": "NcaaGameSimulatorV2"
        },
        "Version 3": {
            "file": "ncaa_simv3.py",
            "class": "NcaaGameSimulatorV3"
        },
        "Version 4": {
            "file": "ncaa_simv4.py",
            "class": "NcaaGameSimulatorV3"  # Note: V4 still uses V3 class name
        }
    }
    
    # Ensure the stats directory exists
    os.makedirs("stats", exist_ok=True)
    
    try:
        # Load the module
        file_path = version_map[version]["file"]
        module_name = file_path.replace(".py", "")
        module = load_module_from_path(module_name, file_path)
        
        # Get the simulator class and instantiate it
        simulator_class = getattr(module, version_map[version]["class"])
        return simulator_class()
    except Exception as e:
        st.error(f"Error loading simulator: {str(e)}")
        st.error("Please ensure the simulator file and team_stats.csv file are present.")
        return None

# Sidebar for version selection
st.sidebar.header("Simulator Settings")
simulator_version = st.sidebar.selectbox(
    "Select Simulator Version",
    ["Version 2", "Version 3", "Version 4"],
    index=2  # Default to Version 4
)

# Load the selected simulator
simulator = load_simulator(simulator_version)

if simulator is None:
    st.stop()

# Load team names for dropdowns
@st.cache_data
def get_team_names(simulator):
    try:
        return sorted(simulator.team_stats.index.tolist())
    except:
        return []

team_names = get_team_names(simulator)

# Main content area
col1, col2 = st.columns(2)

with col1:
    # Game settings
    st.subheader("Game Settings")
    neutral_court = st.checkbox("Neutral Court Game")
    
    if not neutral_court:
        home_team = st.selectbox("Home Team", team_names, index=0)
        away_team = st.selectbox("Away Team", team_names, index=1 if len(team_names) > 1 else 0)
    else:
        team1 = st.selectbox("Team 1", team_names, index=0)
        team2 = st.selectbox("Team 2", team_names, index=1 if len(team_names) > 1 else 0)

    num_simulations = st.slider("Number of Simulations", 1000, 100000, 10000, 1000)

with col2:
    # Advanced settings
    st.subheader("Advanced Settings")
    show_histogram = st.checkbox("Show Score Distribution", value=True)
    show_details = st.checkbox("Show Detailed Statistics", value=True)
    
    # Version-specific features
    if simulator_version in ["Version 3", "Version 4"]:
        st.info(f"Using {simulator_version} with enhanced features including team form tracking and clutch performance metrics.")
    else:
        st.info("Using basic simulation model. Upgrade to Version 3 or 4 for enhanced features.")

# Simulate button
if st.button("Run Simulation"):
    try:
        # Get team names based on court type
        if neutral_court:
            team1_name, team2_name = team1, team2
        else:
            team1_name, team2_name = home_team, away_team

        # Check if teams are the same
        if team1_name == team2_name:
            st.error("Please select different teams for the simulation.")
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
            winner = team1_name if team1_win_pct > team2_win_pct else team2_name
            win_pct = max(team1_win_pct, team2_win_pct)
            st.metric("Win Probability", f"{win_pct:.1f}%", f"{winner}")

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
                
                # Add team strength if available
                try:
                    team1_strength = simulator.calculate_team_strength(simulator.team_stats.loc[team1_name.lower()])
                    st.write(f"Team Strength: {team1_strength:.2f}")
                except:
                    pass

            with stats_col2:
                st.write(f"**{team2_name} Statistics:**")
                st.write(f"Average Score: {team2_avg:.1f} ± {team2_std:.1f}")
                st.write(f"Win Percentage: {team2_win_pct:.1f}%")
                
                # Add team strength if available
                try:
                    team2_strength = simulator.calculate_team_strength(simulator.team_stats.loc[team2_name.lower()])
                    st.write(f"Team Strength: {team2_strength:.2f}")
                except:
                    pass

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
    <p>Data source: Team statistics from KenPom and other sources</p>
</div>
""", unsafe_allow_html=True)

# Version information in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Version Information")
st.sidebar.markdown("""
- **Version 2**: Basic simulation model
- **Version 3**: Enhanced with team form tracking and clutch performance
- **Version 4**: Further improvements with better tempo control and reduced regression to the mean
""")

# Help section in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Help")
st.sidebar.markdown("""
1. Select a simulator version
2. Choose teams to simulate
3. Adjust simulation settings
4. Click "Run Simulation"
""")