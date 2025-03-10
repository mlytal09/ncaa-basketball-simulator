import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time
import base64
from io import BytesIO

# Import simulator
from ncaa_simv3 import NcaaGameSimulatorV3

# Set page configuration
st.set_page_config(
    page_title="NCAA Basketball Game Simulator V3",
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
    st.title("NCAA Basketball Game Simulator V3")
    st.markdown("""
    This app simulates NCAA basketball games using advanced statistics and Monte Carlo simulations.
    Select teams and simulation settings below to see predicted outcomes.
    """)
    
    # Create sidebar for simulation settings
    st.sidebar.header("Simulation Settings")
    
    num_simulations = st.sidebar.slider(
        "Number of Simulations", 
        min_value=1000, 
        max_value=50000, 
        value=10000, 
        step=1000,
        help="More simulations = more accurate results but slower"
    )
    
    # Initialize simulator
    try:
        simulator = NcaaGameSimulatorV3()
        
        # Ensure team_stats is loaded
        if simulator.team_stats is None:
            st.error("Failed to load team statistics. Please check your KenPom credentials.")
            st.stop()
            
    except Exception as e:
        st.error(f"Error initializing simulator: {str(e)}")
        st.stop()
    
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
        
        # Get team options
        team_options = sorted(simulator.team_stats['Team'].unique())
        
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
        
        # Create a button to run simulation
        run_button = st.button("Run Simulation", type="primary")
    
    # Run the simulation when the button is clicked
    if run_button:
        with st.spinner(f"Simulating {num_simulations} games between {team1} and {team2}..."):
            # Show simulation info
            st.info(f"Court: {'Neutral' if neutral_court else f'{team1} home'}")
            
            # Run simulation
            start_time = time.time()
            results = simulator.simulate_game(team1, team2, n_simulations=num_simulations, neutral_site=neutral_court)
            sim_time = time.time() - start_time
            
            # Display results
            st.success(f"Simulation completed in {sim_time:.2f} seconds")
            
            results_col1, results_col2 = st.columns(2)
            
            with results_col1:
                st.subheader("Win Probability")
                win_prob = results['win_probability'] * 100
                loss_prob = (1 - results['win_probability']) * 100
                overtime_prob = results.get('overtime_probability', 0) * 100
                
                # Create win probability chart
                fig, ax = plt.subplots(figsize=(8, 6))
                labels = [f"{team1}", f"{team2}", "Overtime"]
                sizes = [win_prob, loss_prob, overtime_prob]
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                
                ax.bar(labels, sizes, color=colors)
                ax.set_ylabel('Probability (%)')
                ax.set_title('Win Probability')
                for i, v in enumerate(sizes):
                    ax.text(i, v + 1, f"{v:.1f}%", ha='center')
                
                st.pyplot(fig)
                
                # Win probability text summary
                st.write(f"{team1}: {win_prob:.1f}%")
                st.write(f"{team2}: {loss_prob:.1f}%")
                st.write(f"Chance of Overtime: {overtime_prob:.1f}%")
                
                # Confidence rating
                confidence_rating = min(5, max(1, int(abs(win_prob - 50) / 10)))
                confidence_stars = "★" * confidence_rating + "☆" * (5 - confidence_rating)
                st.write(f"Prediction Confidence: {confidence_stars}")
            
            with results_col2:
                st.subheader("Game Analysis")
                
                # Show predicted score
                st.write("**Predicted Score:**")
                st.write(results['predicted_score'])
                
                # Show key factors
                st.write("**Key Factors:**")
                for factor in results['key_factors']:
                    st.write(f"• {factor}")
                
                # Show team styles
                st.write("\n**Team Styles:**")
                st.write(f"{team1}: {simulator.get_team_scoring_style(team1)}")
                st.write(f"{team2}: {simulator.get_team_scoring_style(team2)}")
                
                if not neutral_court:
                    st.write(f"\n**Home Court Advantage:** {simulator.calculate_home_court_advantage(team1):.1f} points")

if __name__ == "__main__":
    main()