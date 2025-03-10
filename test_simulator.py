import unittest
import pandas as pd
import numpy as np
from ncaa_simv3 import NcaaGameSimulatorV3

class TestNcaaSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=== Starting NCAA Basketball Simulator Tests ===\n")
        cls.simulator = NcaaGameSimulatorV3()
        
        # Create sample data for testing
        sample_data = {
            'Team': ['Kansas', 'Duke', 'North Carolina', 'Kentucky'],
            'Conference': ['Big 12', 'ACC', 'ACC', 'SEC'],
            'AdjOE': [115.2, 114.8, 113.5, 112.9],
            'AdjDE': [90.1, 91.2, 92.5, 91.8],
            'Tempo-Adj': [70.5, 68.2, 71.3, 69.8],
            '3P%': [37.5, 35.2, 34.8, 36.1],
            'eFG%': [54.2, 53.8, 52.9, 53.5],
            'TOV%': [16.5, 17.2, 18.1, 17.5],
            'ORB%': [32.5, 31.8, 30.9, 31.5],
            'FTR': [35.2, 34.8, 33.9, 34.5],
            'HomeAdvantage': [3.5, 3.5, 3.5, 3.5]
        }
        
        # Convert to DataFrame
        cls.sample_df = pd.DataFrame(sample_data)
        cls.simulator.team_stats = cls.sample_df
        cls.simulator._update_scoring_styles()
        
        # Test teams
        cls.team1 = "Kansas"
        cls.team2 = "Duke"
        
    def test_kenpom_integration(self):
        """Test KenPom data integration."""
        print("\nTesting KenPom Integration...")
        
        # Test that team stats are loaded
        self.assertIsNotNone(self.simulator.team_stats)
        self.assertGreater(len(self.simulator.team_stats), 0)
        
        # Check required columns exist
        required_columns = ['Team', 'Conference', 'AdjOE', 'AdjDE', 'Tempo-Adj']
        for col in required_columns:
            self.assertIn(col, self.simulator.team_stats.columns)
        
        print("✓ KenPom integration test passed")
    
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation functionality."""
        print("\nTesting Monte Carlo Simulation...")
        
        # Test with a small number of simulations
        n_sims = 100
        try:
            results = self.simulator.simulate_game(self.team1, self.team2, n_simulations=n_sims)
            
            # Check that results contain required keys
            required_keys = [
                'win_probability', 'predicted_score', 'key_factors', 'overtime_probability'
            ]
            for key in required_keys:
                self.assertIn(key, results)
            
            # Check probability values are valid
            self.assertTrue(0 <= results['win_probability'] <= 1)
            self.assertTrue(0 <= results['overtime_probability'] <= 1)
            
            print("✓ Monte Carlo simulation test passed")
            
        except Exception as e:
            self.fail(f"Monte Carlo simulation test failed: {str(e)}")
    
    def test_scoring_styles(self):
        """Test scoring style classification."""
        print("\nTesting Scoring Style Classification...")
        
        team = self.team1
        style = self.simulator.get_team_scoring_style(team)
        self.assertIsNotNone(style)
        self.assertIn(style, ['Fast-paced', 'Balanced', 'Methodical'])
        
        print("✓ Scoring style classification test passed")
    
    def test_home_court_advantage(self):
        """Test home court advantage calculation."""
        print("\nTesting Home Court Advantage...")
        
        home_advantage = self.simulator.calculate_home_court_advantage(self.team1)
        self.assertTrue(2.0 <= home_advantage <= 5.0)
        
        print("✓ Home court advantage test passed")
    
    def test_matchup_analysis(self):
        """Test matchup analysis functionality."""
        print("\nTesting Matchup Analysis...")
        
        results = self.simulator.simulate_game(self.team1, self.team2)
        
        # Check matchup analysis exists
        self.assertIn('win_probability', results)
        self.assertIn('predicted_score', results)
        self.assertIn('key_factors', results)
        
        print("✓ Matchup analysis test passed")
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        print("\nTesting Error Handling...")
        
        # Test invalid team names
        with self.assertRaises(ValueError):
            self.simulator.simulate_game("Invalid Team 1", "Invalid Team 2")
        
        # Test same team matchup
        with self.assertRaises(ValueError):
            self.simulator.simulate_game(self.team1, self.team1)
        
        print("✓ Error handling test passed")

def run_tests():
    """Run all tests and print summary."""
    print("\n=== Starting NCAA Basketball Simulator Tests ===\n")
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNcaaSimulator)
    
    # Run tests
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    return len(result.failures) + len(result.errors) == 0

if __name__ == '__main__':
    success = run_tests()
    if not success:
        print("\n⚠️ Some tests failed. Please review the output above.")
    else:
        print("\n✅ All tests passed successfully!")