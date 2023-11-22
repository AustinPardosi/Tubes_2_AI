import numpy as np
import pickle

class K_Nearest_Neighbor:
    def __init__(self, k):
        self.k = k
        self.points = None

    def euclidean_distance(self, point1, point2):
        diff = np.array(point1) - np.array(point2)
        total = np.sum(diff**2)
        result = np.sqrt(total)
        return result

    def fit(self, points):
        self.points = points

    def predict(self, new_point):
        distances = []
        for category, category_points in self.points.items():
            for point in category_points:
                distance = self.euclidean_distance(point, new_point)
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
            return pickle.load(file)
