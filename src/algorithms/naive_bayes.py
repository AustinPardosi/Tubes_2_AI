import math
import pickle

class NaiveBayes:
    def __init__(self):
        self.columns_name= ["battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g", "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height", "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g", "touch_screen", "wifi"]
        self.nominal_columns = ["blue", "dual_sim", "four_g", "three_g", "touch_screen", "wifi"]
        self.data_count = 0
        self.label_count = {}
        self.feature_model = {}
        self.trained = False

    ### UTILITY METHODS ###
    def is_trained(self):
        return self.trained
    
    def calculate_gauss_probability(self, val, mean, std):
        expo = math.exp(-(math.pow(val - mean, 2) / (2 * math.pow(std, 2))))
        return (1 / (math.sqrt(2 * math.pi) * std)) * expo

    def reset(self):
        self.data_count = 0
        self.label_count = {}
        self.feature_model = {}

    ### LEARNING METHOD ###
    def learn_from_data(self, data):
        # Reset data and set trained as true
        self.reset()
        self.trained = True

        # Store number of data
        self.data_count = data.shape[0]

        # Store number of each label in data
        for label, number in data["price_range"].value_counts().items():
            self.label_count[label] = number
        
        # Gain model for each feature for each label
        for label in self.label_count:
            self.feature_model[label] = {}

            for feature in self.columns_name:
                self.feature_model[label][feature] = {}
                feature_data = data[(data["price_range"] == label)][feature]

                # Numerical feature, use normal distribution probability function. Store mean and std
                if feature not in self.nominal_columns:
                    self.feature_model[label][feature]["mean"] = feature_data.mean()
                    self.feature_model[label][feature]["std"] = feature_data.std()
                # Nominal feature. Store count
                else:
                    for unique in feature_data.unique():
                        self.feature_model[label][feature][unique] = len(feature_data[(feature_data == unique)])

    ### CLASSIFYING METHOD ###
    def calculate_label_probability(self, label, data):
        # Given that label is true, base probability is the probability of that label
        probability = self.label_count[label] / self.data_count

        # Calculate every feature probability given label is true
        for feature in self.columns_name:
            if feature in self.nominal_columns:
                feature_prob = self.feature_model[label][feature][data[feature]] / self.label_count[label]
            else:
                feature_prob = self.calculate_gauss_probability(data[feature],
                                                                 self.feature_model[label][feature]["mean"],
                                                                 self.feature_model[label][feature]["std"])
            probability *= feature_prob
        
        return probability

    def classify(self, data):
        bestLabel = None
        bestProbability = -1

        for label in self.label_count:
            probability = self.calculate_label_probability(label, data)
            if (bestLabel is None) or (probability > bestProbability):
                bestLabel = label
                bestProbability = probability
            
        return bestLabel
    
    ### SAVING AND LOADING METHOD ###
    def dump(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)


    @staticmethod
    def load(filename):
        with open(filename, "rb") as file:
            model = pickle.load(file)
            model.trained = True
            return model