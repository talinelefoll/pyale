import unittest
import os
import sys
import pandas as pd
from unittest.mock import patch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Adjust Python path to include src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from pyale.ale_1d_calcu import calculate_ale_1d
from pyale.ale_1d_viz import ale_1d_plot

class TestALEPlot(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load example data
        data_path = os.path.join(os.path.dirname(__file__), 'var_cars.csv')
        cls.var_cars = pd.read_csv(data_path)

        # List of categorical columns
        categorical_cols = ["vs", "am", "gear", "country"]

        # Apply Label Encoding
        label_encoders = {}
        for col in categorical_cols:
            label_encoders[col] = LabelEncoder()
            cls.var_cars[col] = label_encoders[col].fit_transform(cls.var_cars[col])
        

        cls.X = cls.var_cars.drop(columns=['mpg'])
        cls.y = cls.var_cars['mpg']
        cls.model = LinearRegression()
        cls.model.fit(cls.X, cls.y)
        
        # Define the type and name of the model
        cls.model_type = 'LinearRegression'
        cls.model_name = cls.model

        # Calculate ALE values
        cls.ale_1d_ds = calculate_ale_1d(cls.model_type, cls.model_name, cls.X, cls.y)

    @patch('matplotlib.pyplot.show')
    def test_ale_1d_plot(self, mock_show):
        try:
            ale_1d_plot(
                data=self.var_cars,
                target_col='mpg',
                ale_ds=self.ale_1d_ds,
                categorical_cols=["vs", "am", "gear", "country"],
                centralization='median'
            )
        except Exception as e:
            self.fail(f"ale_1d_plot raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
