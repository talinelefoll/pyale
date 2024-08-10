import unittest
import sys
import os
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Adjust Python path to include src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from pyale.ale_1d_calcu import calculate_ale_1d

class TestALECalculation(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Define the path to the CSV file
        data_path = os.path.join(os.path.dirname(__file__), 'var_cars.csv')
        
        # Load example data
        cls.var_cars = pd.read_csv(data_path)

        # List of categorical columns
        categorical_cols = ["vs", "am", "gear", "country"]

        # Apply Label Encoding
        label_encoders = {}
        for col in categorical_cols:
            if col in cls.var_cars.columns:
                label_encoders[col] = LabelEncoder()
                cls.var_cars[col] = label_encoders[col].fit_transform(cls.var_cars[col])

        # Convert into categorical datatype if necessary
        cls.var_cars = cls.var_cars.astype({'vs': 'category', 'am': 'category', 'gear': 'category', 'country': 'category'})

        # Prepare features and target
        cls.X = cls.var_cars.drop(columns=['mpg'])
        cls.y = cls.var_cars['mpg']
        cls.model = LinearRegression()
        cls.model.fit(cls.X, cls.y)
        
        # Define the type and name of the model
        cls.model_type = 'LinearRegression'
        cls.model_name = cls.model
    
    def test_calculate_ale_1d(self):
        ale_results = calculate_ale_1d(self.model_type, self.model_name, self.X, self.y)
        
        # Check if the result is not None
        self.assertIsNotNone(ale_results)
        
        # Check if the result contains ALE data for each feature
        features = self.X.columns
        for feature in features:
            self.assertIn(f"{feature}__{self.model_type}__ale", ale_results)

if __name__ == '__main__':
    unittest.main()
