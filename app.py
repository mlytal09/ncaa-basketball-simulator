import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import time
import base64
from io import BytesIO
import requests

# Import simulator classes
from ncaa_simv2 import NcaaGameSimulatorV2
from ncaa_simv3 import NcaaGameSimulatorV3
from ncaa_simv4 import NcaaGameSimulatorV4

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
        # Create stats directory if it doesn't exist
        os.makedirs("stats", exist_ok=True)
        
        # Load the team stats directly from GitHub
        st.sidebar.info(f"Loading team stats from GitHub...")
        
        # Try to fetch the data with a timeout
        response = requests.get(github_stats_url, timeout=10)
        if response.status_code == 200:
            # Save to a file in the stats directory
            with open("stats/team_stats.csv", "w") as f:
                f.write(response.text)
            progress_bar.progress(50)
            st.sidebar.success(f"Team stats downloaded successfully")
        else:
            st.sidebar.error(f"Failed to load team stats: HTTP {response.status_code}")
    except Exception as e:
        st.sidebar.error(f"Error downloading team stats: {str(e)}")
    
    progress_bar.progress(75)
    
    # Initialize simulator based on version
    try:
        if simulator_version == "Standard (V2)":
            simulator = NcaaGameSimulatorV2(stats_dir="stats")
            st.sidebar.info("Using Standard simulator (V2)")
        elif simulator_version == "Advanced (V3)":
            simulator = NcaaGameSimulatorV3(stats_dir="stats")
            st.sidebar.info("Using Advanced simulator (V3) with improved accuracy")
        else:
            simulator = NcaaGameSimulatorV4(stats_dir="stats")
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
        
        # Get team options from the simulator
        if hasattr(simulator, 'team_stats') and simulator.team_stats is not None:
            if simulator.team_stats.index.name == 'team_name':
                team_options = sorted(simulator.team_stats.index.tolist())
            else:
                # Fallback if index is not team_name
                team_options = sorted(simulator.team_stats.iloc[:, 0].tolist())
        else:
            # Fallback if team_stats is not available
            team_options = ["Team A", "Team B", "Team C", "Team D", "Team E"]
        
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
        
        # Check for rivalry game if the simulator has that method
        if hasattr(simulator, 'is_rivalry_game') and callable(getattr(simulator, 'is_rivalry_game')):
            try:
                if simulator.is_rivalry_game(team1, team2):
                    st.warning(f"⚡ {team1} vs {team2} is a RIVALRY GAME! Expect the unexpected!")
            except:
                pass
        
        # Create a button to run simulation
        run_button = st.button("Run Simulation", type="primary")
    
    # Run the simulation when the button is clicked
    if run_button:
        with st.spinner(f"Simulating {num_simulations} games between {team1} and {team2}..."):
            # Show simulation info
            st.info(f"Court: {'Neutral' if neutral_court else f'{team1} home'}")
            
            # Run simulations
            start_time = time.time()
            
            try:
                # Use the run_simulation method if available
                if hasattr(simulator, 'run_simulation') and callable(getattr(simulator, 'run_simulation')):
                    results = simulator.run_simulation(team1, team2, neutral_court=neutral_court, num_simulations=num_simulations)
                    
                    # Extract results
                    team1_wins = results.get('team1_wins', 0)
                    team2_wins = results.get('team2_wins', 0)
                    overtime_games = results.get('overtime_games', 0)
                    team1_scores = results.get('team1_scores', [])
                    team2_scores = results.get('team2_scores', [])
                    team1_margins = [team1_scores[i] - team2_scores[i] for i in range(len(team1_scores))]
                else:
                    # Fallback to manual simulation if run_simulation is not available
                    team1_wins = 0
                    team2_wins = 0
                    overtime_games = 0
                    team1_scores = []
                    team2_scores = []
                    team1_margins = []
                    
                    # Create a progress bar for simulations
                    sim_progress = st.progress(0)
                    
                    for i in range(num_simulations):
                        # Update progress every 5% of simulations
                        if i % max(1, num_simulations // 20) == 0:
                            sim_progress.progress(i / num_simulations)
                        
                        # Run simulation
                        result = simulator.simulate_game(team1, team2, neutral_court=neutral_court)
                        
                        # Extract scores and overtime flag
                        if isinstance(result, tuple) and len(result) >= 2:
                            score1 = result[0]
                            score2 = result[1]
                            is_overtime = result[2] if len(result) > 2 else False
                        else:
                            # Fallback if result format is unexpected
                            score1 = result.get('team1_score', 70)
                            score2 = result.get('team2_score', 65)
                            is_overtime = result.get('is_overtime', False)
                        
                        team1_scores.append(score1)
                        team2_scores.append(score2)
                        team1_margins.append(score1 - score2)
                        
                        if score1 > score2:
                            team1_wins += 1
                        elif score2 > score1:
                            team2_wins += 1
                        
                        if is_overtime:
                            overtime_games += 1
                    
                    # Complete the progress bar
                    sim_progress.progress(1.0)
            except Exception as e:
                st.error(f"Error in simulation: {str(e)}")
                st.error(f"Team 1: {team1}, Team 2: {team2}")
                st.stop()
            
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
        
        # Use the simulator's histogram function if available
        if hasattr(simulator, '_generate_score_histogram') and callable(getattr(simulator, '_generate_score_histogram')):
            try:
                # Create simulation_results directory if it doesn't exist
                os.makedirs("simulation_results", exist_ok=True)
                
                # Generate histogram using the simulator's method
                histogram_path = simulator._generate_score_histogram(team1, team2, team1_scores, team2_scores)
                
                # Display the histogram
                st.image(histogram_path)
                
                # Provide download link
                st.markdown(f"[Download Histogram]({histogram_path})", unsafe_allow_html=True)
            except Exception as e:
                # Fallback to manual histogram generation
                st.error(f"Error generating histogram with simulator: {str(e)}")
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
                
                # Provide download link using in-memory buffer
                st.markdown(get_image_download_link(fig, 
                                                  f"{team1}_vs_{team2}.png", 
                                                  "Histogram"), 
                          unsafe_allow_html=True)
        else:
            # Manual histogram generation
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
            
            # Provide download link using in-memory buffer
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