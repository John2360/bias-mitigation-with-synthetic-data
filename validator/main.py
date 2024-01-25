class Validator:
    def __init__(self):
        self.valid_data = None
        self.data_to_validate = None

    def load_valid_data(self, data):
        self.valid_data = data

    def load_data_to_validate(self, data):
        self.data_to_validate = data

    def get_column_min_max(self, column):
        data = self.valid_data[column]
        return data.min(), data.max()
    
    def validate_range(self, column, min_max = None):
        min, max = None, None
        if min_max is None:
            min, max = self.get_column_min_max(column)
        else:
            min, max = min_max

        data = self.data_to_validate[column]
        return (data >= min) & (data <= max)
    
    def get_unique_values(self, column):
        return self.valid_data[column].unique()

    def validate_enum(self, column, enum=None):
        data = self.data_to_validate[column]

        if enum is None:
            enum = self.get_unique_values(column)

        return data.isin(enum)
    
    def validate_type(self, column, type):
        data = self.data_to_validate[column]
        return data.apply(lambda x: isinstance(x, type))
    
    
