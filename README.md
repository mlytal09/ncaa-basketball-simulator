# NCAA Basketball Game Simulator

A web application that simulates NCAA men's basketball games based on team statistics.

## Features

- Simulates games between any NCAA Division I men's basketball teams
- Incorporates team statistics for realistic predictions
- Models home court advantage and conference strength
- Calculates win probabilities, predicted scores, and more
- Generates score distribution visualizations
- Supports both neutral court and home/away games
- Detects rivalry games for more realistic outcomes

## How It Works

The simulator uses a statistical model based on team metrics to predict the outcome of games. It takes into account:

- Offensive and defensive efficiency
- Four Factors (shooting, turnovers, rebounding, free throws)
- Strength of schedule
- Home court advantage
- Conference strength
- Team form and momentum
- And more...

The simulation runs thousands of game simulations to provide win probabilities and score predictions with statistical significance.

## Running Locally

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`

## Data Sources

The simulator uses basketball statistics. You'll need to provide your own CSV file with team data to use the app.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- SciPy
- scikit-learn

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Overview

This simulator uses advanced basketball metrics (adjusted efficiency, four factors, etc.) to simulate NCAA basketball games. It runs 50,000 simulations for each matchup to provide statistical predictions of game outcomes.

Key features:
- Simulates games based on team statistics
- Accounts for home court advantage and conference strength
- Handles neutral court games
- Generates score distributions and visualizations
- Provides win probability and score predictions
- Includes confidence ratings for predictions

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Prepare your data:
   - Place team statistics in CSV format in the main directory or `stats` folder
   - Name your main stats file `team_stats.csv`
   - Include conference information in column B

## Usage

Run the simulator:
```
python ncaa_simv3.py
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
- Column A: Team name (used as the index)
- Column B: Conference name
- AdjO: Adjusted offensive efficiency (points per 100 possessions)
- AdjD: Adjusted defensive efficiency (points allowed per 100 possessions)
- Tempo: Adjusted tempo (possessions per 40 minutes)
- eFG%: Effective field goal percentage
- TOV%: Turnover percentage
- ORB%: Offensive rebound percentage
- FTR: Free throw rate

You can add additional metrics to enhance the simulation model.

## Example

```
==================================================
        NCAA Basketball Game Simulator v3       
==================================================

Is this game on a neutral court? (yes/no): no
Enter the Home team: Missouri
Enter the Away team: Kentucky

Simulating 50000 games between Missouri and Kentucky...
Court: Missouri home

==================================================
                SIMULATION RESULTS
==================================================

Simulations completed: 50000 (91.53 seconds)

Missouri vs Kentucky
Court: Missouri home

Win Probability:
Missouri: 66.0%
Kentucky: 34.0%
Chance of Overtime: 0.0%
Prediction Confidence: ★★★☆☆

Average Score:
Missouri: 79.6 ± 4.4
Kentucky: 77.4 ± 4.8
Margin: 2.2 points (95% confidence interval: 2.1 to 2.2)

Most Common Score:
Missouri 79 - Kentucky 78

Predicted Final:
Missouri 80 - Kentucky 77

Score distribution histogram saved to: simulation_results/Missouri_vs_Kentucky_20250308_122551.png
```

## Extending the Simulator

You can enhance the simulator by:
- Adding more metrics to the CSV file
- Modifying the simulation logic in the `simulate_game` method
- Implementing player-level statistics for more detailed simulations 