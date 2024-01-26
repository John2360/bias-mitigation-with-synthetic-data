from sklearn.metrics import confusion_matrix

class Stats:
    def __init__(self, predictions, labels, positive=1):
        self.predictions = predictions
        self.labels = labels
        self.positive_label = 1
        
        self.true_positives = None
        self.true_negatives = None
        self.false_positives = None
        self.false_negatives = None

    def calculate_accuracy(self):
        conf_matrix = confusion_matrix(y_true=self.labels, y_pred=self.predictions)
        return (conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix.sum()

    def calculate_spd(self, group_a, group_b):
        conf_matrix_group_a = confusion_matrix(y_true=self.labels[group_a], y_pred=self.predictions[group_a])
        conf_matrix_group_b = confusion_matrix(y_true=self.labels[group_b], y_pred=self.predictions[group_b])

        # Calculate probabilities of predicted positive for Group A and Group B
        prob_predicted_positive_group_a = conf_matrix_group_a[1, 1] / (conf_matrix_group_a[1, 1] + conf_matrix_group_a[0, 1])
        prob_predicted_positive_group_b = conf_matrix_group_b[1, 1] / (conf_matrix_group_b[1, 1] + conf_matrix_group_b[0, 1])

        # Calculate Statistical Parity Difference
        return prob_predicted_positive_group_a - prob_predicted_positive_group_b
    
    def calculate_di(self, protected_group, refrence_group):
        # Calculate confusion matrices for the protected and reference groups
        conf_matrix_protected_group = confusion_matrix(y_true=self.labels[protected_group], y_pred=self.predictions[protected_group])
        conf_matrix_reference_group = confusion_matrix(y_true=self.labels[refrence_group], y_pred=self.predictions[refrence_group])

        # Calculate probabilities of predicted positive for the protected and reference groups
        prob_predicted_positive_protected_group = conf_matrix_protected_group[1, 1] / (conf_matrix_protected_group[1, 1] + conf_matrix_protected_group[0, 1])
        prob_predicted_positive_reference_group = conf_matrix_reference_group[1, 1] / (conf_matrix_reference_group[1, 1] + conf_matrix_reference_group[0, 1])

        # Calculate Disparate Impact
        return prob_predicted_positive_protected_group / prob_predicted_positive_reference_group
    
    def calculate_eod(self, protected_group, refrence_group):
        # Calculate confusion matrices for the protected and reference groups
        conf_matrix_protected_group = confusion_matrix(y_true=self.labels[protected_group], y_pred=self.predictions[protected_group])
        conf_matrix_reference_group = confusion_matrix(y_true=self.labels[refrence_group], y_pred=self.predictions[refrence_group])

        # Calculate true positive rates for the protected and reference groups
        true_positive_rate_protected_group = conf_matrix_protected_group[1, 1] / (conf_matrix_protected_group[1, 1] + conf_matrix_protected_group[1, 0])
        true_positive_rate_reference_group = conf_matrix_reference_group[1, 1] / (conf_matrix_reference_group[1, 1] + conf_matrix_reference_group[1, 0])

        # Calculate Equal Opportunity Difference
        return true_positive_rate_protected_group - true_positive_rate_reference_group
    
    def calculate_aaod(self, protected_group, refrence_group):
        # Calculate confusion matrices for the protected and reference groups
        conf_matrix_protected_group = confusion_matrix(y_true=self.labels[protected_group], y_pred=self.predictions[protected_group])
        conf_matrix_reference_group = confusion_matrix(y_true=self.labels[refrence_group], y_pred=self.predictions[refrence_group])

        # Calculate odds for the protected and reference groups
        odds_protected_group = conf_matrix_protected_group[1, 1] / conf_matrix_protected_group[0, 1]
        odds_reference_group = conf_matrix_reference_group[1, 1] / conf_matrix_reference_group[0, 1]

        # Calculate Average Absolute Odds Difference
        return 0.5 * abs(odds_protected_group - odds_reference_group)
    
    def true_positives(self):
        if self.true_positives is not None:
            return self.true_positives
        
        conf_matrix = confusion_matrix(y_true=self.labels, y_pred=self.predictions)
        self.true_positives = conf_matrix[self.positive_label, self.positive_label]
        return self.true_positives

    def true_negatives(self):
        if self.true_negatives is not None:
            return self.true_negatives
        
        conf_matrix = confusion_matrix(y_true=self.labels, y_pred=self.predictions)
        self.true_negatives = conf_matrix[0, 0]
        return self.true_negatives

    def false_positives(self):
        if self.false_positives is not None:
            return self.false_positives
        
        conf_matrix = confusion_matrix(y_true=self.labels, y_pred=self.predictions)
        self.false_positives = conf_matrix[0, self.positive_label]
        return self.false_positives

    def false_negatives(self):
        if self.false_negatives is not None:
            return self.false_negatives
        
        conf_matrix = confusion_matrix(y_true=self.labels, y_pred=self.predictions)
        self.false_negatives = conf_matrix[self.positive_label, 0]
        return self.false_negatives

    def true_positive_rate(self):
        tp = self.true_positives()
        fn = self.false_negatives()
        return tp / (tp + fn)

    def true_negative_rate(self):
        tn = self.true_negatives()
        fp = self.false_positives()
        return tn / (tn + fp)

    def false_positive_rate(self):
        fp = self.false_positives()
        tn = self.true_negatives()
        return fp / (fp + tn)

    def false_negative_rate(self):
        fn = self.false_negatives()
        tp = self.true_positives()
        return fn / (fn + tp)

    def false_discovery_rate(self):
        fp = self.false_positives()
        tp = self.true_positives()
        return fp / (fp + tp)

    def false_omission_rate(self):
        fn = self.false_negatives()
        tp = self.true_positives()
        return fn / (fn + tp)

    def positive_predictive_value(self):
        tp = self.true_positives()
        fp = self.false_positives()
        return tp / (tp + fp)

    def negative_predictive_value(self):
        tn = self.true_negatives()
        fn = self.false_negatives()
        return tn / (tn + fn)

    def rate_of_positive_predictions(self):
        tp = self.true_positives()
        fp = self.false_positives()
        return (tp + fp) / len(self.labels)

    def rate_of_negative_predictions(self):
        tn = self.true_negatives()
        fn = self.false_negatives()
        return (tn + fn) / len(self.labels)

    def accuracy(labels, predictions):
        tn, _, _, tp = confusion_matrix(y_true=labels, y_pred=predictions).ravel()
        return (tn + tp) / len(labels)
    
    def save_metrics(self, file_name='model_metrics.txt', doPrint=True):
        if doPrint:
            print('Calculating metrics...')

        metrics = {
            'True Positives': self.true_positives(),
            'True Negatives': self.true_negatives(),
            'False Positives': self.false_positives(),
            'False Negatives': self.false_negatives(),
            'True Positive Rate': self.true_positive_rate(),
            'True Negative Rate': self.true_negative_rate(),
            'False Positive Rate': self.false_positive_rate(),
            'False Negative Rate': self.false_negative_rate(),
            'False Discovery Rate': self.false_discovery_rate(),
            'False Omission Rate': self.false_omission_rate(),
            'Positive Predictive Value': self.positive_predictive_value(),
            'Negative Predictive Value': self.negative_predictive_value(),
            'Rate of Positive Predictions': self.rate_of_positive_predictions(),
            'Rate of Negative Predictions': self.rate_of_negative_predictions(),
            'Accuracy': self.calculate_accuracy(),
            'Statistical Parity Difference': self.calculate_spd(),
            'Disparate Impact': self.calculate_di(),
            'Equal Opportunity Difference': self.calculate_eod(),
            'Average Absolute Odds Difference': self.calculate_aaod()
        }
        
        with open(file_name, 'w') as file:
            for key, value in metrics.items():
                file.write(f'{key}: {value}\n')
                
                if doPrint:
                    print(f'{key}: {value}')
    