import pandas as pd

class Downsampler: 
    """Downsamples dataset to balance classes."""
    def __init__(self):
        self.data = None   

    def load_data(self, data):
        """Loads data from pandas dataframe."""
        if '_weights' not in data.columns:
            data['_weights'] = 1.0
        self.data = data

    def downsample(self, feature="", majority="", minority="", n_samples=100, upweight=False):
        """Downsamples dataset to balance classes."""
        # if ratio < 0 or ratio > 1:
        #     raise Exception("ratio must be between 0 and 1")

        if majority == "" or minority == "" or feature == "":
            raise Exception("feature, majority, and minority must be specified")
        
        if self.data is None:
            raise Exception("data must be loaded before generating data")
        
        if feature not in self.data.columns:
            raise Exception("feature must be a column in the data")
        
        majority_data = self.data[self.data[feature] == majority]
        minority_data = self.data[self.data[feature] == minority]
        minority_majority_free_data = self.data[(self.data[feature] != majority) & (self.data[feature] != minority)]

        if len(majority_data) < len(minority_data):
            raise Exception("majority class must be less than or equal to minority class")
 
        # n_samples = int(len(minority_data) / (1 - ratio))
        majority_resample = majority_data.sample(n=len(majority_data) - n_samples, replace=True)
        decrease_ratio = n_samples / len(majority_data)
        
        if upweight:
            minority_data['_weights'] = minority_data['_weights'] * decrease_ratio

        resampled_data = pd.concat([majority_resample, minority_data, minority_majority_free_data])
        resampled_data = resampled_data.sample(frac=1).reset_index(drop=True)

        if not upweight:
            return resampled_data.loc[:, resampled_data.columns != '_weights']

        return resampled_data
