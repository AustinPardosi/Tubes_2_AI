import numpy as np
import pickle

class K_Nearest_Neighbor:
    def __init__(self, k):
        self.k = k
        self.points = None
        self.categories = None
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
        return result

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
        
# data = pd.read_csv("D:\Vs Code\Tubes_2_AI\data\data_train.csv")
# obj = K_Nearest_Neighbor(3)
# obj.fit(data)

# validation = pd.read_csv("D:\Vs Code\Tubes_2_AI\data\data_validation.csv")

# true = 0
# for i in range(len(validation)):
#     if obj.predict(validation.iloc[i].drop("price_range")) == validation.iloc[i]["price_range"]:
#         true += 1

# accuracy = true / len(validation)
# print("Accuracy:", accuracy)