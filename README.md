# NCAA Basketball Game Simulator

A web application that simulates NCAA men's basketball games using advanced analytics and team statistics.

## Features

* Simulates games between any NCAA Division I men's basketball teams
* Incorporates team statistics for realistic predictions
* Models home court advantage and conference strength
* Calculates win probabilities, predicted scores, and more
* Generates score distribution visualizations
* Supports both neutral court and home/away games
* Detects rivalry games for more realistic outcomes
* Streamlit web interface for easy interaction

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mlytal09/ncaa-basketball-simulator.git
cd ncaa-basketball-simulator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Interface

Run the Streamlit app:
```bash
streamlit run app.py
```

This will open the web interface in your default browser where you can:
- Select teams to simulate
- Choose between home/away or neutral court games
- Adjust the number of simulations
- View detailed statistics and visualizations

### Command Line

You can also run simulations directly from the command line:
```bash
python ncaa_simv4.py
```

## Data Format

The simulator expects a CSV file with team statistics in the `stats` directory. The file should include:

- Team names
- Conference information
- Adjusted offensive and defensive efficiency
- Four Factors (eFG%, TOV%, ORB%, FTR)
- Additional metrics (optional)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
