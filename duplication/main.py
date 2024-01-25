class DuplicationGenerator: 
    """Generates new data by duplicating samples from the majority class and changing their label to the minority class."""
    def __init__(self):
        self.data = None   

    def load_data(self, data):
        """Loads data from pandas dataframe."""
        self.data = data

    def generate(self, feature="", majority="", minority="", n_samples=100):
        """Generates new data by duplicating samples from the majority class and changing their label to the minority class."""
        if n_samples < 1:
            raise Exception("n_samples must be greater than 0")

        if majority == "" or minority == "" or feature == "":
            raise Exception("feature, majority, and minority must be specified")
        
        if self.data is None:
            raise Exception("data must be loaded before generating data")
        
        if feature not in self.data.columns:
            raise Exception("feature must be a column in the data")
        
        majority_data = self.data[self.data[feature] == majority]
        sample = majority_data.sample(n=n_samples, replace=True)
        sample[feature] = minority

        return sample
