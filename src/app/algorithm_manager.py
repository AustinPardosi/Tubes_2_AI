# Class AlgorithmManager as an interface between App and the algorithm

from algorithms.KNN import K_Nearest_Neighbor
from algorithms.naive_bayes import NaiveBayes
import pandas as pd

class AlgorithmManager:
    def __init__(self):
        self.naive_bayes = NaiveBayes()
        self.knn = K_Nearest_Neighbor(33)

    # Methods to train using algorithm
    def train_knn(self, filename, load_flag):
        if (load_flag):
            self.knn = K_Nearest_Neighbor.load(filename)
        else:
            train_data = pd.read_csv(filename)
            self.knn.fit(train_data)
    
    def train_naive_bayes(self, filename, load_flag):
        if (load_flag):
            self.naive_bayes = NaiveBayes.load(filename)
        else:
            train_data = pd.read_csv(filename)
            self.naive_bayes.learn_from_data(train_data)
    
    # Methods to check if algorithm trained or not
    def is_knn_trained(self):
        return self.knn.is_trained()

    def is_naive_bayes_trained(self):
        return self.naive_bayes.is_trained()
    
    # Methods to do labelling with algorithm
    def classify_with_knn(self, test_filename, output_filename):
        test_data = pd.read_csv(f"../test/{test_filename}")

        result_data = pd.DataFrame()
        result_data["id"] = test_data["id"]
        test_data.drop("id", axis=1, inplace=True)

        result_label = []
        for i in range(0, len(test_data)):
            result_label.append(self.knn.predict(test_data.iloc[i]))
        result_data["price_range"] = result_label

        result_data.to_csv(f"../result/{output_filename}", index=False)

    def classify_with_naive_bayes(self, test_filename, output_filename):
        test_data = pd.read_csv(f"../test/{test_filename}")

        result_data = pd.DataFrame()
        result_data["id"] = test_data["id"]
        test_data.drop("id", axis=1, inplace=True)

        result_label = []
        for i in range(0, len(test_data)):
            result_label.append(self.naive_bayes.classify(test_data.iloc[i]))
        result_data["price_range"] = result_label

        result_data.to_csv(f"../result/{output_filename}", index=False)

    # Methods to save the model
    def dump_knn(self, output_filename):
        self.knn.dump(f"../model/{output_filename}")

    def dump_naive_bayes(self, output_filename):
        self.naive_bayes.dump(f"../model/{output_filename}")

    # Methods to train using algorithm
    def test_knn_acc(self):
        validation_data = pd.read_csv("../data/data_validation.csv")
        true = 0

        for i in range(len(validation_data)):
            if self.knn.predict(validation_data.iloc[i].drop("price_range")) == validation_data.iloc[i]["price_range"]:
                true += 1

        print(f"Jumlah label yang sesuai dengan data: {true}")
        print(f"Persentase akurasi algoritma KNN: {(true/len(validation_data))*100}%")

    def test_naive_bayes_acc(self):
        validation_data = pd.read_csv("../data/data_validation.csv")
        true = 0

        for i in range(0, len(validation_data)):
            if (self.naive_bayes.classify(validation_data.iloc[i]) == validation_data.iloc[i]["price_range"]):
                true += 1

        print(f"Jumlah label yang sesuai dengan data: {true}")
        print(f"Persentase akurasi algoritma Naive Bayes: {(true/len(validation_data))*100}%")
        
        