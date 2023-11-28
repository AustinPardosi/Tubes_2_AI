# File to implement KNN and Naive Bayes Algorithm using scikit-learn
# This file will not be integrated with main.py and stand as its own
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the training dataset
train_data = pd.read_csv("../data/data_train.csv")

# Load the validation dataset
validation_data = pd.read_csv("../data/data_validation.csv")

# Split the training data into features and target
X_train = train_data.drop("price_range", axis=1)
y_train = train_data["price_range"]

# Split the validation data into features and target
X_validation = validation_data.drop("price_range", axis=1)
y_validation = validation_data["price_range"]

# KNN Implementation
knn_model = KNeighborsClassifier(n_neighbors=33)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_validation)
knn_accuracy = accuracy_score(y_validation, knn_predictions)
print(f"Akurasi KNN: {knn_accuracy}")

# Naive Bayes Implementation
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_validation)
nb_accuracy = accuracy_score(y_validation, nb_predictions)
print(f"Akurasi Naive Bayes: {nb_accuracy}")