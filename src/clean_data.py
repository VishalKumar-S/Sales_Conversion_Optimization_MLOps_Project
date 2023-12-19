import pandas as pd

class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def clean_data(self)->pd.DataFrame:
        # Apply cleaning operations to the DataFrame
        self.data = self.drop_rows(self.data, 'Impressions', 2000000)
        self.data = self.drop_rows(self.data, 'Spent', 500)
        self.data = self.drop_columns(self.data, ['ad_id', 'xyz_campaign_id', 'fb_campaign_id', 'gender', 'interest', 'age'])
        self.data['CPC'] = self.calculate_cpc(self.data)
        self.data = self.impute_missing_values(self.data, ['CPC'])
        
        # Return the cleaned DataFrame
        return self.data

    def drop_rows(self, data, column, threshold):
        return data.loc[data[column] <= threshold]

    def drop_columns(self, data, columns):
        return data.drop(columns, axis=1)

    def calculate_cpc(self, data):
        data['CPC'] = data['Spent'] / data['Clicks']
        return data['CPC']

    def impute_missing_values(self, data, columns):
        data[columns] = data[columns].fillna(0)
        return data
