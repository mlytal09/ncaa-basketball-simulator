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

# Import simulator classes
try:
    from ncaa_simv2 import NcaaGameSimulatorV2
    from ncaa_simv3 import NcaaGameSimulatorV3
    from ncaa_simv4 import NcaaGameSimulatorV3 as NcaaGameSimulatorV4
    AVAILABLE_VERSIONS = ["V2", "V3", "V4"]
except ImportError as e:
    st.error(f"Error importing simulator classes: {e}")
    st.error("Please ensure all simulator files are in the correct location.")
    AVAILABLE_VERSIONS = []

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

# Initialize simulator based on version
@st.cache_resource(show_spinner=False)
def get_simulator(version):
    """Initialize and return the appropriate simulator version"""
    try:
        if version == "V2":
            return NcaaGameSimulatorV2()
        elif version == "V3":
            return NcaaGameSimulatorV3()
        elif version == "V4":
            return NcaaGameSimulatorV4()
        else:
            st.error(f"Unknown simulator version: {version}")
            return None
    except Exception as e:
        st.error(f"Error initializing simulator: {e}")
        return None

# Get team names from simulator
def get_team_names(simulator_instance):
    """Get sorted list of team names from simulator"""
    if simulator_instance is None or not hasattr(simulator_instance, 'team_stats'):
        return []
    
    try:
        # Convert index to list for safety
        return sorted([str(name) for name in simulator_instance.team_stats.index.tolist()])
    except Exception as e:
        st.error(f"Error getting team names: {e}")
        return []

# Main function
def main():
    # Add a title and description
    st.title("NCAA Basketball Game Simulator 🏀")
    st.markdown("""
    This app simulates NCAA basketball games based on team statistics.
    Select teams and simulation settings below to see predicted outcomes.
    
    Data is loaded from team_stats.csv which includes team statistics and conference information.
    """)
    
    # Create sidebar for simulation settings
    st.sidebar.header("Simulation Settings")
    
    # Add option to choose simulator version
    version_descriptions = {
        "V2": "Basic simulator with fundamental team metrics",
        "V3": "Enhanced model with rivalry detection and team form tracking",
        "V4": "Latest version with improved scoring model and tempo control"
    }
    
    simulator_version = st.sidebar.radio(
        "Simulator Version",
        AVAILABLE_VERSIONS,
        index=min(2, len(AVAILABLE_VERSIONS)-1),  # Default to V4 if available
        help="Choose which version of the simulator to use"
    )
    
    # Show version description
    if simulator_version in version_descriptions:
        st.sidebar.info(version_descriptions[simulator_version])
    
    # Initialize simulator based on selection
    with st.spinner(f"Loading {simulator_version} simulator..."):
        simulator = get_simulator(simulator_version)
    
    if simulator is None:
        st.error("Failed to initialize simulator. Please check the logs for details.")
        return
    
    # Get team names for dropdowns
    team_names = get_team_names(simulator)
    if not team_names:
        st.error("No team data available. Please check that team_stats.csv is properly formatted and located in the stats directory.")
        return
    
    # Number of simulations slider
    num_simulations = st.sidebar.slider(
        "Number of Simulations", 
        min_value=1000, 
        max_value=50000, 
        value=10000, 
        step=1000,
        help="More simulations = more accurate results but slower"
    )
    
    # Create columns for the main UI
    col1, col2 = st.columns(2)
    
    # Game setup
    with col1:
        st.subheader("Game Setup")
        court_type = st.radio(
            "Court Type", 
            ["Home/Away", "Neutral Court"], 
            index=0
        )
        neutral_court = (court_type == "Neutral Court")
        
        if neutral_court:
            # Neutral court game
            team1 = st.selectbox(
                "Select Team 1:",
                options=team_names,
                index=0
            )
            
            team2 = st.selectbox(
                "Select Team 2:",
                options=team_names,
                index=min(1, len(team_names)-1)
            )
        else:
            # Home/Away game
            team1 = st.selectbox(
                "Select Home Team:",
                options=team_names,
                index=0
            )
            
            team2 = st.selectbox(
                "Select Away Team:",
                options=team_names,
                index=min(1, len(team_names)-1)
            )
    
    # Run simulation button
    with col2:
        st.subheader("Run Simulation")
        st.write("Click the button below to run the simulation with the selected settings.")
        
        # Add some spacing
        st.write("")
        st.write("")
        
        # Check for rivalry game if using V3 or V4
        if simulator_version in ["V3", "V4"] and hasattr(simulator, 'is_rivalry_game'):
            try:
                if simulator.is_rivalry_game(team1, team2):
                    st.warning(f"⚡ {team1} vs {team2} is a RIVALRY GAME! Expect the unexpected!")
            except Exception as e:
                st.info(f"Could not check for rivalry: {e}")
        
        # Create a button to run simulation
        run_button = st.button("Run Simulation", type="primary")
    
    # Run the simulation when the button is clicked
    if run_button:
        if team1 == team2:
            st.error("Please select different teams for the simulation.")
            return
            
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
            
            progress_bar = st.progress(0)
            
            for i in range(num_simulations):
                try:
                    # Try the new V3/V4 interface first (returns 3 values)
                    score1, score2, is_overtime = simulator.simulate_game(team1, team2, neutral_court)
                    if is_overtime:
                        overtime_games += 1
                except ValueError:
                    # Fall back to old interface (returns 2 values)
                    score1, score2 = simulator.simulate_game(team1, team2, neutral_court)
                except Exception as e:
                    st.error(f"Error during simulation: {e}")
                    break
                
                team1_scores.append(score1)
                team2_scores.append(score2)
                team1_margins.append(score1 - score2)
                
                if score1 > score2:
                    team1_wins += 1
                elif score2 > score1:
                    team2_wins += 1
                
                # Update progress bar every 5% of simulations
                if i % max(1, num_simulations // 20) == 0:
                    progress_bar.progress(i / num_simulations)
            
            # Complete progress bar
            progress_bar.progress(1.0)
            
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
            score_pairs = [(int(team1_scores[i]), int(team2_scores[i])) for i in range(num_simulations)]
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
            
            # Confidence rating (V3/V4 feature)
            if simulator_version in ["V3", "V4"]:
                confidence_rating = min(5, max(1, int(abs(team1_wins - team2_wins) / (num_simulations * 0.1))))
                confidence_stars = "★" * confidence_rating + "☆" * (5 - confidence_rating)
                st.write(f"Prediction Confidence: {confidence_stars}")
        
        with results_col2:
            st.subheader("Score Prediction")
            
            # Show average scores
            st.write(f"{team1}: {team1_avg:.1f} ± {team1_std:.1f}")
            st.write(f"{team2}: {team2_avg:.1f} ± {team2_std:.1f}")
            st.write(f"Margin: {margin:.1f} points")
            
            # Confidence interval (V3/V4 feature)
            if simulator_version in ["V3", "V4"]:
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
        bins1 = range(int(min(team1_scores))-5, int(max(team1_scores))+5)
        ax1.hist(team1_scores, bins=bins1, color='blue', alpha=0.7)
        ax1.axvline(np.mean(team1_scores), color='red', linestyle='dashed', linewidth=2)
        ax1.set_xlabel('Points')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{team1} Score Distribution')
        
        # Team 2 histogram
        bins2 = range(int(min(team2_scores))-5, int(max(team2_scores))+5)
        ax2.hist(team2_scores, bins=bins2, color='green', alpha=0.7)
        ax2.axvline(np.mean(team2_scores), color='red', linestyle='dashed', linewidth=2)
        ax2.set_xlabel('Points')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{team2} Score Distribution')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add download link for the histogram
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{team1}_vs_{team2}_{timestamp}.png"
        st.markdown(get_image_download_link(fig, filename, "histogram"), unsafe_allow_html=True)
        
        # Add a divider
        st.markdown("---")
        
        # Additional insights section
        st.subheader("Additional Insights")
        
        # Create columns for additional insights
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            # Score breakdown
            st.write("**Score Breakdown:**")
            st.write(f"Highest {team1} score: {max(team1_scores):.1f}")
            st.write(f"Lowest {team1} score: {min(team1_scores):.1f}")
            st.write(f"Highest {team2} score: {max(team2_scores):.1f}")
            st.write(f"Lowest {team2} score: {min(team2_scores):.1f}")
            
            # Blowout percentage
            blowout_threshold = 15
            team1_blowouts = sum(1 for margin in team1_margins if margin > blowout_threshold)
            team2_blowouts = sum(1 for margin in team1_margins if margin < -blowout_threshold)
            st.write(f"**Blowout Chance (15+ point margin):**")
            st.write(f"{team1} blowout: {team1_blowouts/num_simulations*100:.1f}%")
            st.write(f"{team2} blowout: {team2_blowouts/num_simulations*100:.1f}%")
        
        with insight_col2:
            # Close game percentage
            close_threshold = 5
            close_games = sum(1 for margin in team1_margins if abs(margin) <= close_threshold)
            st.write(f"**Close Game Chance (margin ≤ 5 points):**")
            st.write(f"{close_games/num_simulations*100:.1f}%")
            
            # Overtime chance
            st.write(f"**Overtime Chance:**")
            st.write(f"{overtime_games/num_simulations*100:.1f}%")
            
            # Team form if available in V3/V4
            if simulator_version in ["V3", "V4"] and hasattr(simulator, 'get_team_form'):
                try:
                    team1_form = simulator.get_team_form(team1)
                    team2_form = simulator.get_team_form(team2)
                    st.write("**Team Form:**")
                    st.write(f"{team1}: {team1_form}")
                    st.write(f"{team2}: {team2_form}")
                except Exception:
                    pass

# Run the main function
if __name__ == "__main__":
    main()