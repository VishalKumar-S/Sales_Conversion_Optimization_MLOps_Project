# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def clean_data(self):
        # Apply cleaning operations to the DataFrame
        self.data.drop(self.data.loc[self.data['Impressions'] > 2000000].index, inplace=True)
        self.data.drop(self.data.loc[self.data['Spent'] > 500].index, inplace=True)

        self.data.drop(['ad_id', 'xyz_campaign_id', 'fb_campaign_id', 'gender', 'interest', 'age'], axis=1, inplace=True)

        # Calculate Cost per Click (CPC)
        self.data['CPC'] = self.data['Spent'] / self.data['Clicks']

        # Impute missing values with 0 for specified columns (CPC)
        columns_to_impute = ['CPC']
        self.data[columns_to_impute] = self.data[columns_to_impute].fillna(0)

        # Return the cleaned DataFrame
        return self.data

