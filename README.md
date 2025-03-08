# NCAA Basketball Game Simulator

A Python-based simulator for NCAA men's basketball games using KenPom statistics to predict outcomes.

## Overview

This simulator uses advanced basketball metrics from KenPom (adjusted efficiency, four factors, etc.) to simulate NCAA basketball games. It runs 50,000 simulations for each matchup to provide statistical predictions of game outcomes.

Key features:
- Simulates games based on KenPom statistics
- Accounts for home court advantage
- Handles neutral court games
- Generates score distributions and visualizations
- Provides win probability and score predictions

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Prepare your data:
   - Place KenPom statistics in CSV format in the `stats` folder
   - Name your main stats file `kenpom_stats.csv`
   - See `sample_kenpom_stats.csv` for the expected format

## Usage

Run the simulator:
```
python ncaa_simulator.py
```

The program will:
1. Ask if the game is on a neutral court
2. Prompt for the home team name
3. Prompt for the away team name
4. Run 50,000 simulations
5. Display results in the console
6. Save a histogram of score distributions to the `simulation_results` folder

## Data Format

The simulator expects a CSV file with the following columns:
- Team: Team name (used as the index)
- AdjO: Adjusted offensive efficiency (points per 100 possessions)
- AdjD: Adjusted defensive efficiency (points allowed per 100 possessions)
- Tempo: Adjusted tempo (possessions per 40 minutes)
- eFG_Pct: Effective field goal percentage
- TO_Pct: Turnover percentage
- OR_Pct: Offensive rebound percentage
- FTRate: Free throw rate
- HomeAdvantage: Home court advantage in points

You can add additional KenPom metrics to enhance the simulation model.

## Example

```
==================================================
        NCAA Basketball Game Simulator        
==================================================

Is this game on a neutral court? (yes/no): no
Enter the home team: Gonzaga
Enter the away team: Baylor

Simulating 50000 games between Gonzaga and Baylor...
Court: Gonzaga home

==================================================
              SIMULATION RESULTS               
==================================================
Simulations completed: 50000 (5.23 seconds)

Gonzaga vs Baylor
Court: Gonzaga home

Win Probability:
Gonzaga: 65.3%
Baylor: 34.6%
Tie: 0.1%

Average Score:
Gonzaga: 81.5 ± 6.2
Baylor: 76.3 ± 5.8

Most Common Score:
Gonzaga 82 - Baylor 76

Predicted Final:
Gonzaga 82 - Baylor 76

Score distribution histogram saved to: simulation_results/Gonzaga_vs_Baylor_20250308_051054.png
```

## Extending the Simulator

You can enhance the simulator by:
- Adding more KenPom metrics to the CSV file
- Modifying the simulation logic in the `simulate_game` method
- Implementing player-level statistics for more detailed simulations 