import numpy as np
import pickle

class K_Nearest_Neighbor:
    def __init__(self, k):
        self.k = k
        self.points = None
        self.categories = None
        self.normalize_params = {}
        self.columns_to_drop = ["blue", "clock_speed", "dual_sim", "fc", "four_g", "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "sc_h", "sc_w", "talk_time", "three_g", "touch_screen", "wifi"]
        self.trained = False

    ### UTILITY METHODS ###
    def is_trained(self):
        return self.trained
    
    def calculate_euclidean_distance(self, point1, point2):
        diff = np.array(point1) - np.array(point2)
        total = np.sum(diff**2)
        result = np.sqrt(total)
        return result

    def reset(self):
        self.points = None
        self.categories = None

    ### LEARNING METHOD ###
    def fit(self, data):
        self.reset()
        self.trained = True

        # Drop unuseful columns
        data.drop(columns = self.columns_to_drop, inplace=True)

        # # Normalize data for KNN
        column_names = data.columns.tolist()
        column_names.remove("price_range")
        for column in column_names:
            max = data[column].max()
            min = data[column].min()

            data[column] = (data[column] - min) / (max - min)

            self.normalize_params[column] = {}
            self.normalize_params[column]["max"] = max
            self.normalize_params[column]["min"] = min

        # Learn from data by organizing it into points for each category.
        self.points = {}
        self.categories = data["price_range"].unique()

        for category in self.categories:
            category_data = data[data["price_range"] == category]
            features_list = [
                row.drop("price_range").tolist() for _, row in category_data.iterrows()
            ]
            self.points[category] = features_list

    ### CLASSIFYING METHOD ###
    def predict(self, new_point):
        # Drop unuseful columns
        new_point = new_point.drop(labels = self.columns_to_drop)

        # Normalize data
        for column in new_point.keys():
            new_point[column] = (new_point[column] - self.normalize_params[column]["min"]) / (self.normalize_params[column]["max"] - self.normalize_params[column]["min"])

        # Predict the category of a new point using KNN.
        distances = []
        for category, category_points in self.points.items():
            for point in category_points:
                distance = self.calculate_euclidean_distance(point, new_point)
                distances.append([distance, category])
        distances = np.array(distances)
        sorted_distances = distances[distances[:, 0].argsort()]
        k_nearest_categories = sorted_distances[: self.k, 1]
        unique_categories, counts = np.unique(k_nearest_categories, return_counts=True)
        result = unique_categories[np.argmax(counts)]
        return int(result)

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